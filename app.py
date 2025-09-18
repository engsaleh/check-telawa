# app.py
"""
Quran Reciter App
مهمّة: تطبيق لاختبار التلاوة (جزء 30).
"""

import os
import time
import tempfile
import shutil
import random
import requests
import re
import unicodedata
import json
from difflib import SequenceMatcher
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from transformers import pipeline
from pydub import AudioSegment
from jiwer import wer

# ------------- إعداد التطبيق -------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "change_this_in_prod")

# حد حجم التحميل (10 ميجابايت)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ------------- إعداد النموذج -------------
# يمكنك تغييره إلى "tarteel-ai/whisper-tiny-ar-quran" إن لم يتوفر GPU
MODEL_NAME = "tarteel-ai/whisper-base-ar-quran"

# اختَر الجهاز التلقائي (GPU إذا كان متاحاً)
import torch

device = 0 if torch.cuda.is_available() else -1

# تحميل بايبلاين ASR (قد يستغرق وقتًا عند التشغيل الأول)
print("Loading ASR pipeline:", MODEL_NAME, "device:", device)
asr_pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device=device)
print("ASR pipeline loaded.")

# ------------- إعدادات API للقرآن الكريم -------------
API_BASE_URL = "https://api.quran.com/api/v4"


# متغيرات تخزين مؤقت للسور (Cache)
_cached_surahs = None
_cache_timestamp = None
CACHE_DURATION = 3600  # ساعة واحدة


# ------------- وظائف API للقرآن الكريم -------------
def clean_html_tags(text):
    """إزالة علامات HTML من النص."""
    clean_text = re.sub(r"<sup.*?</sup>", "", text)
    clean_text = re.sub(r"<[^>]+>", "", clean_text)
    return clean_text.strip()


def get_all_surahs():
    """جلب قائمة جميع السور من API مع نظام تخزين مؤقت."""
    global _cached_surahs, _cache_timestamp

    # فحص التخزين المؤقت
    if _cached_surahs and _cache_timestamp:
        if (time.time() - _cache_timestamp) < CACHE_DURATION:
            return _cached_surahs, None

    url = f"{API_BASE_URL}/chapters"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        chapters = response.json().get("chapters", [])

        # تحويل البيانات إلى تنسيق مناسب
        surahs = [(ch["id"], ch["name_arabic"]) for ch in chapters]

        # حفظ في التخزين المؤقت
        _cached_surahs = surahs
        _cache_timestamp = time.time()

        return surahs, None

    except requests.exceptions.RequestException as e:
        print(f"خطأ في جلب السور: {e}")
        # في حالة الخطأ، ارجع للقائمة اليدوية لجزء عمّ
        return (
            get_juz30_surahs_fallback(),
            f"تم استخدام القائمة المحلية بسبب خطأ في الشبكة: {e}",
        )


def get_juz30_surahs():
    """جلب سور جزء عمّ فقط من API."""
    all_surahs, error = get_all_surahs()
    if error:
        return all_surahs, error

    # تصفية السور لجزء عمّ (السور من 78 إلى 114)
    juz30_surahs = [(id, name) for id, name in all_surahs if 78 <= id <= 114]
    return juz30_surahs, None


def get_juz30_surahs_fallback():
    """قائمة احتياطية لسور جزء عمّ في حالة عدم توفر الإنترنت."""
    return [
        (78, "النبأ"),
        (79, "النازعات"),
        (80, "عبس"),
        (81, "التكوير"),
        (82, "الانفطار"),
        (83, "المطففين"),
        (84, "الانشقاق"),
        (85, "البروج"),
        (86, "الطارق"),
        (87, "الأعلى"),
        (88, "الغاشية"),
        (89, "الفجر"),
        (90, "البلد"),
        (91, "الشمس"),
        (92, "الليل"),
        (93, "الضحى"),
        (94, "الشرح"),
        (95, "التين"),
        (96, "العلق"),
        (97, "القدر"),
        (98, "البينة"),
        (99, "الزلزلة"),
        (100, "العاديات"),
        (101, "القارعة"),
        (102, "التكاثر"),
        (103, "العصر"),
        (104, "الهمزة"),
        (105, "الفيل"),
        (106, "قريش"),
        (107, "الماعون"),
        (108, "الكوثر"),
        (109, "الكافرون"),
        (110, "النصر"),
        (111, "المسد"),
        (112, "الإخلاص"),
        (113, "الفلق"),
        (114, "الناس"),
    ]


def get_surah_verses(surah_id):
    """جلب آيات سورة معينة."""
    url = f"{API_BASE_URL}/chapters/{surah_id}/verses"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        verses = data.get("verses", [])
        return verses, None
    except requests.exceptions.RequestException as e:
        print(f"خطأ في جلب آيات السورة {surah_id}: {e}")
        return [], f"خطأ في جلب آيات السورة: {e}"


def get_ayah_from_quran_api(surah_id, ayah_number):
    """جلب نص آية معينة من API الرئيسي (Quran.com)."""
    url = f"{API_BASE_URL}/verses/by_key/{surah_id}:{ayah_number}"
    params = {"fields": "text_imlaei"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        verse = data.get("verse", {})
        text = verse.get("text_imlaei", "")
        return clean_html_tags(text)
    except Exception as e:
        print(f"خطأ في API الرئيسي: {e}")
    return ""


def get_ayah_text(surah_id, ayah_number):
    """جلب نص الآية مع نظام احتياطي متعدد."""
    # أولاً: جرب API الرئيسي
    text = get_ayah_from_quran_api(surah_id, ayah_number)
    if text:
        return text
    # وإلا لا يوجد نص متاح
    return ""


# ------------- وظائف مساعدة -------------
def remove_diacritics(text):
    """إزالة التشكيل - للاستخدام في حالات خاصة فقط"""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def normalize_text_for_compare(text):
    """تطبيع النص مع الاحتفاظ بالتشكيل للمقارنة الدقيقة"""
    if not text:
        return ""

    # لا نزيل التشكيل، فقط نزيل علامات الترقيم والأرقام
    text = re.sub(
        r"[^\w\s\u0600-\u06FF\u064B-\u065F\u0610-\u061A\u06D6-\u06ED]", "", text
    )  # احتفظ بالحروف العربية والتشكيل
    text = unicodedata.normalize("NFKC", text)  # تطبيع Unicode للتشكيل
    return " ".join(text.split())  # تنظيف المسافات


def color_diff_html(ref_text, hyp_text):
    """
    تلوين كلمة-بكلمة باستخدام opcodes من SequenceMatcher.
    الكلمات المتطابقة تصبح باللون الأخضر، الكلمات المستبدلة بالأحمر،
    الإدخالات بالبرتقالي، الحذوفات بخط مائل.
    """
    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    s = SequenceMatcher(None, ref_words, hyp_words)
    parts = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            for w in ref_words[i1:i2]:
                parts.append(f"<span class='word correct'>{w}</span>")
        elif tag == "replace":
            for w in hyp_words[j1:j2]:
                parts.append(f"<span class='word wrong'>{w}</span>")
        elif tag == "insert":
            for w in hyp_words[j1:j2]:
                parts.append(f"<span class='word insert'>{w}</span>")
        elif tag == "delete":
            for w in ref_words[i1:i2]:
                parts.append(f"<span class='word deleted'>{w}</span>")
    return " ".join(parts)


def convert_to_wav(src_path):
    """
    تحويل أي ملف صوتي وارد (webm/ogg/mp3) إلى WAV مونو 16kHz باستخدام pydub + ffmpeg.
    إرجاع مسار ملف WAV الجديد.
    """
    basename = os.path.splitext(os.path.basename(src_path))[0]
    out_wav = os.path.join(tempfile.gettempdir(), f"{basename}_conv.wav")
    audio = AudioSegment.from_file(src_path)
    # نُنقّح إلى mono و 16kHz (تحسين للتعرف)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(out_wav, format="wav")
    return out_wav


# ------------- منطق تقييم التلاوة (محدّث) -------------
def evaluate_recitation(file_path, surah_id, ayah_number):
    """
    1) تحويل الملف إلى WAV إن لزم
    2) تشغيل ASR لإخراج النص
    3) جلب النص المرجعي من API
    4) مقارنة، حساب WER، تجهيز HTML ملون وإخراج رسالة تشجيع
    """
    # تأكد من صلاحية المدخلات (أساسية)
    try:
        wav_path = convert_to_wav(file_path)
    except Exception as e:
        print("Audio convert error:", e)
        # إذا التحويل فشل، جرّب إرسال الملف الأصلي مباشرة إلى الأنبوب
        wav_path = file_path

    # 1) استدعاء نموذج التعرف على الكلام
    asr_result = asr_pipe(wav_path)
    hypothesis = asr_result.get("text", "").strip()

    # 2) نص المرجع من API
    reference_raw = get_ayah_text(surah_id, ayah_number)
    if not reference_raw:
        reference_raw = ""

    # 3) تطبيع النصوص للمقارنة وحساب WER
    ref_norm = normalize_text_for_compare(reference_raw)
    hyp_norm = normalize_text_for_compare(hypothesis)

    # حساب WER باستخدام jiwer
    try:
        error = wer(ref_norm, hyp_norm)
    except Exception:
        error = 1.0

    # 4) توليد HTML ملون
    colored = color_diff_html(ref_norm, hyp_norm)

    # 5)
    if error <= 0.10:
        feedback = "ممتاز"
    elif error <= 0.30:
        feedback = "انتبه لبعض الأخطاء"
    else:
        feedback = "جرب مرة أخرى!"

    return {
        "hypothesis": hypothesis,
        "reference_raw": reference_raw,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "wer": error,
        "colored_html": colored,
        "feedback": feedback,
        "wav_path": wav_path,
    }


# ------------- مسارات الويب (Routes) -------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    # جلب قائمة السور
    surahs, api_error = get_juz30_surahs()

    if api_error:
        flash(f"تحذير: {api_error}", "warning")

    if request.method == "POST":
        # الاستمارة تأتي مع حقول: surah, ayah, audio (ملف)
        try:
            surah = int(request.form.get("surah"))
            ayah = int(request.form.get("ayah"))
        except Exception:
            flash("اختر سورة ورقم آية صالحين", "error")
            return redirect(url_for("index"))

        audio_file = request.files.get("audio")
        if audio_file is None or audio_file.filename == "":
            flash("لم يتم إرفاق تسجيل صوتي", "error")
            return redirect(url_for("index"))

        # نحفظ الملف مؤقتًا ثم نعالجه
        tmp_dir = tempfile.mkdtemp(prefix="qkids_")
        try:
            saved_path = os.path.join(tmp_dir, audio_file.filename)
            audio_file.save(saved_path)

            # تقييم التلاوة
            result = evaluate_recitation(saved_path, surah, ayah)
        finally:
            # نزيل المجلد المؤقت وأي ملف داخله
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

    return render_template("index.html", surahs=surahs, result=result)


@app.route("/get_ayah", methods=["GET"])
def get_ayah_api():
    """API بسيط لجلب نص الآية (يستخدمه الجافاسكربت في الواجهة)."""
    surah = request.args.get("surah")
    ayah = request.args.get("ayah")
    if not surah or not ayah:
        return jsonify({"text": ""}), 400

    text = get_ayah_text(surah, ayah)
    return jsonify({"text": text or ""})


@app.route("/get_surah_info/<int:surah_id>", methods=["GET"])
def get_surah_info(surah_id):
    """جلب معلومات السورة وعدد آياتها."""
    url = f"{API_BASE_URL}/chapters/{surah_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        chapter = data.get("chapter", {})
        return jsonify(
            {
                "id": chapter.get("id"),
                "name_arabic": chapter.get("name_arabic"),
                "verses_count": chapter.get("verses_count"),
                "revelation_place": chapter.get("revelation_place"),
            }
        )
    except Exception as e:
        print(f"خطأ في جلب معلومات السورة: {e}")
        return jsonify({"error": "فشل في جلب معلومات السورة"}), 500


@app.route("/api/surahs", methods=["GET"])
def api_all_surahs():
    """API لجلب جميع السور."""
    surahs, error = get_all_surahs()
    if error:
        return jsonify({"surahs": surahs, "error": error}), 200
    return jsonify({"surahs": surahs})


# ------------- تشغيل التطبيق -------------
if __name__ == "__main__":
    # اختبار الاتصال بالAPI عند بدء التشغيل
    print("اختبار الاتصال بـ API القرآن الكريم...")
    surahs, error = get_juz30_surahs()
    if error:
        print(f"تحذير: {error}")
        print(f"تم تحميل {len(surahs)} سورة من القائمة المحلية")
    else:
        print(f"تم تحميل {len(surahs)} سورة من API بنجاح")

    app.run(host="0.0.0.0", port=5000, debug=True)

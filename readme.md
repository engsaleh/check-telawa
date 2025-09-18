# فاحص التلاوة - جزء عمّ

تطبيق ويب يهدف إلى مساعدة المستخدمين على مراجعة تلاوتهم لجزء عمّ من القرآن الكريم. يقوم المستخدم باختيار السورة ورقم الآية، ثم يرفع تسجيلاً صوتيًا لتلاوته. يقوم التطبيق بعد ذلك بتحليل التسجيل ومقارنته بالنص القرآني الصحيح، ويقدم ملاحظات مرئية على الأخطاء المحتملة ونسبة دقة التلاوة.

## الميزات

*   **اختيار السورة والآية:** قائمة منسدلة لاختيار السورة (جزء عمّ) وحقل لإدخال رقم الآية.
*   **عرض نص الآية:** يعرض نص الآية المختار كمساعد قبل تسجيل التلاوة.
*   **تحميل التسجيل الصوتي:** يدعم تحميل ملفات صوتية (MP3, WAV, M4A, FLAC).
*   **التعرف التلقائي على الكلام (ASR):** يستخدم نموذج Whisper العربي الخاص بالقرآن الكريم لتحويل التلاوة الصوتية إلى نص.
*   **مقارنة النصوص:** يقارن النص المكتشف بالنص القرآني المرجعي.
*   **تصحيح الأخطاء المرئي:** يلون الكلمات الصحيحة والخاطئة والمحذوفة والمضافة لتسهيل تحديد الأخطاء.
*   **مؤشر دقة التلاوة WER:** يعرض نسبة خطأ الكلمات (Word Error Rate).
*   **ملاحظات تشجيعية:** يقدم تغذية راجعة بسيطة بناءً على دقة التلاوة.

## الإعداد والتشغيل

### المتطلبات الأساسية

تأكد من تثبيت الأدوات التالية على نظامك:

*   **Python 3.8+**
*   **pip** (مدير حزم بايثون)
*   **FFmpeg**: ضروري لمعالجة الملفات الصوتية بواسطة مكتبة `pydub`.
    *   **على Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **على Fedora:** `sudo dnf install ffmpeg`
    *   **على macOS (باستخدام Homebrew):** `brew install ffmpeg`
    *   **على Windows:** قم بتنزيل وتثبيت `ffmpeg` من [ffmpeg.org](https://ffmpeg.org/download.html) وتأكد من إضافة مساره إلى متغيرات البيئة (PATH).

### خطوات التثبيت

1.  **استنساخ المستودع Clone the repository:**
    
    ```bash
    git clone https://github.com/YourUsername/QuranReciterChecker.git
    cd QuranReciterChecker
    ```
(غيّر `YourUsername/QuranReciterChecker` إلى المسار الفعلي لمستودعك)
    
2.  **إنشاء بيئة افتراضية Virtual Environment وتفعيلها (مستحسن):**
    
    ```bash
    python -m venv venv
    # على Windows
    .\venv\Scripts\activate
    # على macOS/Linux
    source venv/bin/activate
```
    
3.  **تثبيت المكتبات المطلوبة:**
    ```bash
    pip install -r requirements.txt
    ```

    **ملاحظة لمستخدمي GPU /NVIDIA CUDA:**
    إذا كان لديك بطاقة رسوميات NVIDIA وتريد استخدامها لتسريع عملية التعرف على الكلام (ASR)، قد تحتاج إلى تثبيت `torch` مع دعم CUDA يدويًا. قم بزيارة [pytorch.org](https://pytorch.org/get-started/locally/) للحصول على الأمر الصحيح بناءً على نظامك وإصدار CUDA.
    مثال (لـ CUDA 11.8):
    
    ```bash
    pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    ```
    ثم أعد تثبيت `transformers` لضمان التوافق:
    ```bash
    pip install transformers==4.30.2 accelerate==0.21.0
```
    
4.  **تعيين مفتاح سري لتطبيق Flask:**
    `FLASK_SECRET` مطلوب لجلسات Flask (flash messages). يمكنك تعيينه كمتغير بيئة.
    
    ```bash
    # على macOS/Linux
    export FLASK_SECRET="super_secret_key_change_me_in_prod"
    # على Windows (Command Prompt)
    set FLASK_SECRET="super_secret_key_change_me_in_prod"
    # على Windows (PowerShell)
    $env:FLASK_SECRET="super_secret_key_change_me_in_prod"
    ```
    **ملاحظة:** استخدم مفتاح سري قوي في بيئات الإنتاج

### تشغيل التطبيق

بمجرد الانتهاء من خطوات التثبيت، يمكنك تشغيل التطبيق:

```bash
python app.py
```



بالتأكيد! سأقوم بإنشاء ملفات `requirements.txt` و `README.md` لمشروعك "فاحص التلاوة".

**ملاحظات حول مشروعك:**
*   **الواجهة الأمامية:** تستخدم HTML و CSS وجزءًا من JavaScript (للتعامل مع API لجلب الآيات ومعلومات السورة).
*   **الواجهة الخلفية:** تستخدم Python مع إطار عمل Flask.
*   **التعرف على الكلام (ASR):** تعتمد على مكتبة `transformers` من Hugging Face، وتحديدًا نموذج Whisper العربي للقرآن الكريم.
*   **معالجة الصوت:** تستخدم `pydub` (التي تتطلب `ffmpeg` خارجيًا).
*   **مقارنة النصوص:** تستخدم `difflib` (مكتبة بايثون قياسية) و `jiwer` لحساب WER.
*   **جلب بيانات القرآن:** تعتمد على `requests` للتواصل مع `api.quran.com`.

---

### ملف `requirements.txt`

هذا الملف يحدد جميع المكتبات التي يحتاجها مشروعك لكي يعمل بشكل صحيح.

```
Flask==2.3.2
transformers==4.30.2
torch==2.0.1
torchaudio==2.0.2
SoundFile==0.12.1
accelerate==0.21.0
pydub==0.25.1
jiwer==2.3.0
requests==2.31.0
```

**ملاحظات هامة بخصوص `requirements.txt`:**
*   **`torch` و `torchaudio`:** أرقام الإصدارات قد تحتاج للتعديل بناءً على إعداد بيئتك، خاصة إذا كنت تستخدم CUDA (GPU). يمكنك تثبيتها بدون أرقام إصدارات محددة (`torch`, `torchaudio`) والسماح لـ `pip` باختيار الأحدث المتوافق، أو تحديد الإصدارات التي تعمل لديك.
*   **FFmpeg:** `pydub` تتطلب تثبيت FFmpeg على نظام التشغيل نفسه، ولا يمكن تضمينه في `requirements.txt` لأنه ليس مكتبة بايثون. سأذكر هذا في ملف `README.md`.
*   **`SoundFile`:** غالبًا ما يكون ضروريًا لـ `torchaudio` للتعامل مع الملفات الصوتية.
*   **`accelerate`:** مطلوب بواسطة `transformers` لبعض التحسينات.

---

### ملف `README.md`

هذا الملف سيوفر نظرة عامة على مشروعك، كيفية إعداده، وتشغيله.

```markdown
# 📖 فاحص التلاوة - جزء عمّ

تطبيق ويب يهدف إلى مساعدة المستخدمين على مراجعة تلاوتهم لجزء عمّ من القرآن الكريم. يقوم المستخدم باختيار السورة ورقم الآية، ثم يرفع تسجيلاً صوتيًا لتلاوته. يقوم التطبيق بعد ذلك بتحليل التسجيل ومقارنته بالنص القرآني الصحيح، ويقدم ملاحظات مرئية على الأخطاء المحتملة ونسبة دقة التلاوة.

## ✨ الميزات

*   **اختيار السورة والآية:** قائمة منسدلة لاختيار السورة (جزء عمّ) وحقل لإدخال رقم الآية.
*   **عرض نص الآية:** يعرض نص الآية المختار كمساعد قبل تسجيل التلاوة.
*   **تحميل التسجيل الصوتي:** يدعم تحميل ملفات صوتية (MP3, WAV, M4A, FLAC).
*   **التعرف التلقائي على الكلام (ASR):** يستخدم نموذج Whisper العربي الخاص بالقرآن الكريم لتحويل التلاوة الصوتية إلى نص.
*   **مقارنة النصوص:** يقارن النص المكتشف بالنص القرآني المرجعي.
*   **تصحيح الأخطاء المرئي:** يلون الكلمات الصحيحة والخاطئة والمحذوفة والمضافة لتسهيل تحديد الأخطاء.
*   **مؤشر دقة التلاوة (WER):** يعرض نسبة خطأ الكلمات (Word Error Rate).
*   **ملاحظات تشجيعية:** يقدم تغذية راجعة بسيطة بناءً على دقة التلاوة.

## 🚀 كيفية الإعداد والتشغيل

### المتطلبات الأساسية

تأكد من تثبيت الأدوات التالية على نظامك:

*   **Python 3.8+**
*   **pip** (مدير حزم بايثون)
*   **FFmpeg**: ضروري لمعالجة الملفات الصوتية بواسطة مكتبة `pydub`.
    *   **على Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    *   **على Fedora:** `sudo dnf install ffmpeg`
    *   **على macOS (باستخدام Homebrew):** `brew install ffmpeg`
    *   **على Windows:** قم بتنزيل وتثبيت `ffmpeg` من [ffmpeg.org](https://ffmpeg.org/download.html) وتأكد من إضافة مساره إلى متغيرات البيئة (PATH).

### خطوات التثبيت

1.  **استنساخ المستودع (Clone the repository):**
    ```bash
    git clone https://github.com/YourUsername/QuranReciterChecker.git
    cd QuranReciterChecker
```
    (غيّر `YourUsername/QuranReciterChecker` إلى المسار الفعلي لمستودعك)

2.  **إنشاء بيئة افتراضية (Virtual Environment) وتفعيلها (مستحسن):**
    ```bash
    python -m venv venv
    # على Windows
    .\venv\Scripts\activate
    # على macOS/Linux
    source venv/bin/activate
    ```

3.  **تثبيت المكتبات المطلوبة:**
    ```bash
    pip install -r requirements.txt
    ```

    **ملاحظة لمستخدمي GPU (NVIDIA CUDA):**
    إذا كان لديك بطاقة رسوميات NVIDIA وتريد استخدامها لتسريع عملية التعرف على الكلام (ASR)، قد تحتاج إلى تثبيت `torch` مع دعم CUDA يدويًا. قم بزيارة [pytorch.org](https://pytorch.org/get-started/locally/) للحصول على الأمر الصحيح بناءً على نظامك وإصدار CUDA.
    مثال (لـ CUDA 11.8):
    ```bash
    pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    ```
    ثم أعد تثبيت `transformers` لضمان التوافق:
    ```bash
    pip install transformers==4.30.2 accelerate==0.21.0
    ```

4.  **تعيين مفتاح سري لتطبيق Flask:**
    `FLASK_SECRET` مطلوب لجلسات Flask (flash messages). يمكنك تعيينه كمتغير بيئة.
    ```bash
    # على macOS/Linux
    export FLASK_SECRET="super_secret_key_change_me_in_prod"
    # على Windows (Command Prompt)
    set FLASK_SECRET="super_secret_key_change_me_in_prod"
    # على Windows (PowerShell)
    $env:FLASK_SECRET="super_secret_key_change_me_in_prod"
    ```
    **ملاحظة:** استخدم مفتاحًا سريًا قويًا في بيئات الإنتاج!

### تشغيل التطبيق

بمجرد الانتهاء من خطوات التثبيت، يمكنك تشغيل التطبيق:

```bash
python app.py
```

سيتم تشغيل التطبيق على العنوان `http://127.0.0.1:5000/` (أو `http://localhost:5000/`). افتح هذا العنوان في متصفح الويب الخاص بك.



## التقنيات المستخدمة

*   **الواجهة الخلفية**: Python 3, Flask
*   **التعرف على الكلام ASR- Hugging Face Transformers **: نموذج `tarteel-ai/whisper-base-ar-quran`)
*   **معالجة الصوت:** pydub
*   **مقارنة النصوص:** jiwer, Python's `difflib`
*   **الواجهة الأمامية:** HTML5, CSS3, JavaScript
*   **خطوط الويب:** Google Fonts (خط Amiri)
*   **API بيانات القرآن:** api.quran.com



## المساهمة

أرحب بالمساهمات التحسينية على المشروع، إذا كان لديك أي اقتراحات أو تحسينات، فلا تتردد في فتح issue أو إرسال pull request.


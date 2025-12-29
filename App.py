import os
import tempfile
import numpy as np
import faiss
import nltk
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import gradio as gr
import time

# ==================== تهيئة النظام ====================
class FlowRAGSystem:
    def __init__(self):
        self.model = None
        self.index = None
        self.chunks = None
        self.current_file = None
        self.is_ready = False
    
    def initialize(self):
        """تهيئة النظام"""
        try:
            # تحميل موارد NLTK
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            except:
                pass
            
            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.is_ready = True
            return True
        except Exception as e:
            return f"❌ خطأ في تحميل النموذج: {str(e)}"
    
    def process_pdf(self, pdf_file):
        """معالجة ملف PDF"""
        try:
            self.current_file = pdf_file.name
            
            # حفظ الملف المؤقت
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                pdf_path = tmp_file.name
            
            # قراءة PDF
            reader = PdfReader(pdf_path)
            pages_data = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        'page': i + 1,
                        'text': text.strip()
                    })
            
            if not pages_data:
                os.unlink(pdf_path)
                return "❌ لم يتم العثور على نص في الملف"
            
            # تقسيم النص
            self.chunks = []
            for page in pages_data:
                words = page['text'].split()
                
                # تقسيم إلى أجزاء 200 كلمة مع تداخل 40
                chunk_size = 200
                overlap = 40
                
                start = 0
                while start < len(words):
                    end = start + chunk_size
                    chunk_words = words[start:end]
                    
                    if chunk_words:
                        self.chunks.append({
                            'text': ' '.join(chunk_words),
                            'page': page['page'],
                            'word_count': len(chunk_words)
                        })
                    
                    start += chunk_size - overlap
            
            # إنشاء embeddings
            if len(self.chunks) > 0:
                chunk_texts = [chunk['text'] for chunk in self.chunks]
                embeddings = self.model.encode(
                    chunk_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # بناء الفهرس
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings)
            else:
                os.unlink(pdf_path)
                return "❌ لم يتم إنشاء أي أجزاء نصية"
            
            # تنظيف الملف المؤقت
            os.unlink(pdf_path)
            
            return f"✅ تم معالجة المستند بنجاح!\n📊 {len(pages_data)} صفحة → {len(self.chunks)} جزء نصي"
            
        except Exception as e:
            return f"❌ خطأ في معالجة PDF: {str(e)}"
    
    def search(self, query, top_k=3):
        """بحث في المستند"""
        if not self.is_ready or self.index is None:
            return "❌ يرجى معالجة مستند أولاً"
        
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    
                    # تحديد لون التشابه
                    similarity_score = float(score)
                    if similarity_score >= 0.5:
                        sim_color = "#28a745"  # أخضر
                        sim_text = "ممتاز"
                    elif similarity_score >= 0.3:
                        sim_color = "#ffc107"  # أصفر
                        sim_text = "جيد"
                    else:
                        sim_color = "#dc3545"  # أحمر
                        sim_text = "ضعيف"
                    
                    results.append(f"""
                    <div style="background: #f8f9fa; border-radius: 10px; padding: 1.5rem; 
                    margin: 1rem 0; border-left: 5px solid {sim_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h4 style="margin-top: 0;">🏆 النتيجة #{i+1}</h4>
                        <p style="margin-bottom: 0.5rem;">
                            <span style="color: {sim_color}; font-weight: bold;">التشابه: {score*100:.1f}% ({sim_text})</span> | 
                            📖 الصفحة: {chunk['page']} | 
                            🔢 الكلمات: {chunk['word_count']}
                        </p>
                        <hr style="margin: 0.5rem 0;">
                        <p>{chunk['text'][:400]}...</p>
                    </div>
                    """)
            
            if not results:
                return "❌ لم أجد نتائج ذات صلة في المستند"
            
            return f"<h3>🔍 تم العثور على {len(results)} نتيجة:</h3>" + "".join(results)
            
        except Exception as e:
            return f"❌ خطأ في البحث: {str(e)}"

# ==================== إنشاء النظام ====================
rag_system = FlowRAGSystem()
init_result = rag_system.initialize()

# ==================== واجهة Gradio ====================
with gr.Blocks(title="🤖 نظام RAG الذكي للمستندات", theme=gr.themes.Soft()) as demo:
    
    # العنوان
    gr.Markdown("""
    # 🤖 نظام RAG الذكي للمستندات
    ### بحث دلالي متقدم في ملفات PDF - يدعم العربية والإنجليزية
    """)
    
    # منطقة التنبيهات
    if init_result is not True:
        gr.Warning(f"⚠️ {init_result}")
    else:
        gr.Info("✅ النظام جاهز للاستخدام")
    
    with gr.Row():
        with gr.Column(scale=2):
            # قسم رفع الملف
            with gr.Group():
                gr.Markdown("## 📁 رفع ومعالجة المستند")
                file_input = gr.File(
                    label="اختر ملف PDF",
                    file_types=[".pdf"],
                    type="binary"
                )
                process_btn = gr.Button("🚀 معالجة المستند", variant="primary")
                process_output = gr.Markdown(label="حالة المعالجة")
            
            # قسم البحث
            with gr.Group():
                gr.Markdown("## 💬 اسأل عن المستند")
                question_input = gr.Textbox(
                    label="اكتب سؤالك هنا",
                    placeholder="مثال: ما هي حالة التدفق؟ أو What is flow state?",
                    lines=3
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=5, value=3,
                        label="عدد النتائج"
                    )
                    search_btn = gr.Button("🔍 ابحث في المستند", variant="primary")
                
                search_output = gr.HTML(label="نتائج البحث")
        
        with gr.Column(scale=1):
            # الشريط الجانبي
            with gr.Group():
                gr.Markdown("## 💡 أسئلة سريعة")
                
                example_questions = [
                    "ما هي حالة التدفق؟",
                    "What is flow state?",
                    "ما هي عناصر التجربة المثلى؟",
                    "كيف يحقق الإنسان السعادة في العمل؟",
                    "ما هو دور التركيز في التدفق؟"
                ]
                
                for question in example_questions:
                    gr.Button(
                        question,
                        size="sm",
                    ).click(
                        fn=lambda q=question: q,
                        inputs=[],
                        outputs=[question_input]
                    )
            
            with gr.Group():
                gr.Markdown("## 🎯 نصائح البحث")
                gr.Markdown("""
                **لأفضل النتائج:**
                
                • استخدم مصطلحات محددة  
                • جرب اللغتين (عربي/إنجليزي)  
                • اطرح أسئلة واضحة  
                
                **مثال:**  
                ✅ "ما هي خصائص flow state؟"  
                ❌ "اشرح لي"
                """)
            
            with gr.Group():
                gr.Markdown("## 📊 معلومات النظام")
                status_text = gr.Markdown("📄 لم يتم معالجة أي مستند بعد")
                
                # تحديث حالة النظام
                def update_status():
                    if rag_system.current_file:
                        file_info = f"📄 الملف: {rag_system.current_file}"
                        if rag_system.chunks:
                            chunks_info = f" | 📊 الأجزاء: {len(rag_system.chunks)}"
                            if rag_system.index:
                                vectors_info = f" | 🧮 المتجهات: {rag_system.index.ntotal}"
                                return file_info + chunks_info + vectors_info
                            return file_info + chunks_info
                        return file_info
                    return "📄 لم يتم معالجة أي مستند بعد"
                
                status_display = gr.Markdown(update_status())
    
    # نصائح إضافية
    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📚 عن النظام")
            gr.Markdown("""
            **التقنيات المستخدمة:**
            
            • 🤖 **Sentence Transformers** - نماذج embedding متعددة اللغات  
            • ⚡ **FAISS** - بحث سريع في المتجهات  
            • 📄 **PyPDF** - معالجة ملفات PDF  
            • 🌐 **Gradio** - واجهة مستخدم تفاعلية
            """)
        
        with gr.Column():
            gr.Markdown("### 🌍 الدعم اللغوي")
            gr.Markdown("""
            **اللغات المدعومة:**
            
            • العربية - البحث والنتائج  
            • الإنجليزية - البحث والنتائج  
            • الفرنسية، الإسبانية، الألمانية - البحث الأساسي
            
            **المميزات:**  
            ✓ بحث دلالي ذكي  
            ✓ نتائج مرتبة حسب الصلة  
            ✓ دعم ملفات كبيرة
            """)
    
    # تذييل الصفحة
    gr.Markdown("---")
    gr.Markdown("""
    <div style="text-align: center; color: #666;">
        <p>🤖 نظام RAG للمستندات | إصدار HuggingFace Spaces</p>
        <p>تقنية: FAISS + Sentence Transformers + Gradio | يدعم العربية والإنجليزية</p>
    </div>
    """)
    
    # ==================== معالجة الأحداث ====================
    def process_file(file):
        if file is None:
            return "⚠️ يرجى اختيار ملف PDF أولاً"
        
        result = rag_system.process_pdf(file)
        return result
    
    def search_query(question, top_k):
        if not question:
            return "⚠️ يرجى إدخال سؤال"
        
        return rag_system.search(question, int(top_k))
    
    # ربط الأحداث
    process_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[process_output]
    ).then(
        fn=update_status,
        inputs=[],
        outputs=[status_display]
    )
    
    search_btn.click(
        fn=search_query,
        inputs=[question_input, top_k_slider],
        outputs=[search_output]
    )
    
    # معالجة ضغط Enter في حقل السؤال
    question_input.submit(
        fn=search_query,
        inputs=[question_input, top_k_slider],
        outputs=[search_output]
    )

# ==================== تشغيل التطبيق ====================
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False

'''Request: dùng để lấy dữ liệu từ form (POST/GET).
Render_template: dùng để render HTML template.
Summarizer: là class định nghĩa trong inference.py để thực hiện tóm tắt văn bản.
'''
from flask import Flask, request, render_template
from model.inference import Summarizer

app = Flask(__name__)
'''Name sẽ là __main__ khi chạy trực tiếp file này'''

summarizer = Summarizer()  # Load model khi khởi động
'''Tạo đối tượng summarizer từ class Summarizer để sử dụng trong ứng dụng Flask'''\
'''
- Xác định một đường dẫn route (đường dẫn URL) /
- Methods=['GET', 'POST'] cho phép nhận cả yêu cầu GET và POST.
    - GET: để hiển thị form ban đầu. Cho phép người dùng nhập văn bản cần tóm tắt.
    - POST: để nhận dữ liệu từ form khi người dùng gửi văn bản cần tóm tắt (ấn nút Submit).
'''
@app.route('/', methods=['GET', 'POST'])
def index():
    '''Biến summary để lưu kết quả tóm tắt, 
    input_text để lưu văn bản nhập từ người dùng'''
    summary = ""
    input_text = ""
    '''Kiểm tra nếu người dùng gửi yêu cầu POST (ấn nút Submit trên form), gửi 
    dữ liệu từ form'''

    if request.method == 'POST':
        # Lấy nội dung văn bản người dùng đã nhập trong <textarea name="text">.
        input_text = request.form.get('text', '')
        summary_length = request.form.get('summary_length', 60, type=int)
        # Lấy độ dài tóm tắt người dùng chọn (bằng thanh trượt).
        if input_text:
            summary = summarizer.summarize(input_text, max_length=summary_length)
        '''Nếu người dùng có nhập văn bản, gọi hàm summarize của đối tượng summarizer để tạo bản tóm tắt
        max_length là độ dài tối đa của tóm tắt, do người dùng chọn từ thanh trượt.'''

    return render_template('index.html', summary=summary)

'''Trả lại giao diện index.html, đồng thời truyền các biến summary và input_text để hiển thị kết quả và 
giữ lại nội dung cũ.'''

if __name__ == '__main__':
    app.run(debug=True)

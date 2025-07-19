# Mô hình **PEGASUS**, **T5**, **BART**

## 1. Tổng quan 
### a. Mô hình Pegasus:
- Mô hình Pegasus là một mô hình Transformer encoder - decoder được Google phát triển chuyên cho bài toán tóm tắt ngữ nghĩa (text summarization). Cấu trúc cơ bản của nó tương tự như kiến trúc Transformer chuẩn, gồm bộ mã hóa (encoder) và bộ giải mã (decoder). Mỗi lớp encoder/decoder Pegasus bao gồm cơ chế **multi-head self-attention** và **feed-forward** giống như Transformer gốc. Pegasus sử dụng **mã hóa vị trí sinusoidai (positional encoding sinusodial)** như trong Transformer gốc. Các thông số chính của Pegasus base là Chiều không gian embedding d = 1024, chiều mạng nơ-ron ẩn FNN = 4096, số heads attention = 16 và mỗi phần encoder/decoder có 16 lớp. 

> [!CAUTION]
Nếu chưa nắm được kiến trúc của mô hình Transformer thì có thể đến trang blog này để đọc và nghiên cứu trước: https://pbcquoc.github.io/transformer/


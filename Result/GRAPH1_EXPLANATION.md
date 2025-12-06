# Graph 1 Explanation: Total Accuracy Comparison

## Cấu trúc Graph mới (dễ hiểu hơn)

Graph 1 hiện tại được chia thành **2 subplots riêng biệt**:

### Subplot 1 (Phía trên): Skip-gram Comparison
- **Trục X**: Các configurations (1ep-300d, 1ep-600d, 3ep-300d, 3ep-600d, 10ep-300d, 10ep-600d)
- **Trục Y**: Total Accuracy
- **2 bars cho mỗi configuration**:
  - Bar bên trái (màu đỏ): **HS Skip-gram**
  - Bar bên phải (màu vàng): **NS Skip-gram**
- **So sánh**: Dễ dàng thấy HS Skip-gram vs NS Skip-gram ở cùng một configuration

### Subplot 2 (Phía dưới): CBOW Comparison
- **Trục X**: Các configurations (tương tự)
- **Trục Y**: Total Accuracy
- **2 bars cho mỗi configuration**:
  - Bar bên trái (màu xanh lá): **HS CBOW**
  - Bar bên phải (màu xanh dương): **NS CBOW**
- **So sánh**: Dễ dàng thấy HS CBOW vs NS CBOW ở cùng một configuration

## Cách đọc Graph

### Ví dụ: Configuration "1ep-600d"

**Subplot 1 (Skip-gram)**:
- HS Skip-gram bar: Accuracy = 0.455 (45.5%)
- NS Skip-gram bar: Accuracy = 0.583 (58.3%)
- **Kết luận**: NS Skip-gram tốt hơn HS Skip-gram ở config này

**Subplot 2 (CBOW)**:
- HS CBOW bar: Accuracy = 0.347 (34.7%)
- NS CBOW bar: Accuracy = 0.470 (47.0%)
- **Kết luận**: NS CBOW tốt hơn HS CBOW ở config này

## Lợi ích của cách hiển thị mới

1. **Rõ ràng hơn**: Mỗi subplot chỉ so sánh 2 giá trị (HS vs NS) cho cùng một model
2. **Dễ so sánh**: Bars đặt cạnh nhau nên dễ thấy sự khác biệt
3. **Tránh nhầm lẫn**: Không còn 4 bars cùng lúc gây khó hiểu
4. **Có giá trị trên bars**: Hiển thị chính xác accuracy value trên mỗi bar

## Lưu ý

- Nếu một configuration không có data (ví dụ NS không có 1ep-300d), bar sẽ có giá trị = 0
- Các configurations được sắp xếp theo thứ tự: epochs trước, dimension sau
- Tất cả configurations của cả HS và NS đều được hiển thị để dễ so sánh


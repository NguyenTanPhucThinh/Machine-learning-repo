# Machine Learning Project

## Mô Tả Bài Toán

Dự án này nhằm giải quyết bài toán nhân diện nhân vật anime bằng cách sử dụng các mô hình học máy hiện đại. 

### Mục Tiêu Chính
- Xây dựng và huấn luyện mô hình dự đoán/phân loại
- Đạt được độ chính xác cao trên tập dữ liệu
- [Thêm mục tiêu cụ thể khác của bạn]
### Pipeline
```mermaid
flowchart TD
    A([🖼️ Ảnh đầu vào]) --> B

    subgraph S1["📦 Giai đoạn 1 — Thu thập & Tiền xử lý"]
        B[Thu thập dữ liệu\nGrabber Booru] --> C
        C[Lọc ảnh thủ công\nLoại ảnh không thỏa điều kiện] --> D
        D[Face detection & localization\nLoại background noise] --> E
        E[Resize & Normalize\n448×448 · normalize -1 to 1]
    end

    E --> F

    subgraph S2["🧠 Giai đoạn 2 — Huấn luyện mô hình"]
        F[ViT-Base model\nwd-vit-tagger-v3] --> G
        G[Transfer learning\nFine-tune · 45 nhân vật · 7k ảnh] --> H
        H[Lưu checkpoint\nbest_vit_model.pth] --> I & J
        I[Mô hình phân loại\nAccuracy / F1-Score]
        J[Trích xuất đặc trưng\nTạo embedding gallery]
    end

    I & J --> K

    subgraph S3["⚡ Giai đoạn 3 — Suy luận thời gian thực"]
        K([Ảnh query]) --> L
        L[Tiền xử lý\nMTCNN detect · crop · expand 20–30%] --> M
        M[ViT backbone frozen\nTrích xuất + L2-normalize CLS token] --> N & O
        N[Cosine similarity\nmax sim query vs gallery]
        O[Đầu phân loại\nXác suất softmax]
        N & O --> P{max_sim ≥ τ = 0.657?}
        P -->|Không| Q([❌ UNKNOWN])
        P -->|Có| R[Weighted fusion\nα · gallery + 1-α · cls_prob]
        R --> S([✅ Kết quả nhận diện\nchar_id + điểm tin cậy])
    end

    style S1 fill:#0d3b2e,stroke:#2fa882,color:#c8ede5
    style S2 fill:#1e1040,stroke:#6b5fbe,color:#d4cfee
    style S3 fill:#161624,stroke:#555570,color:#d0d0d8
    style Q fill:#7a2c18,stroke:#b04428,color:#f5c4b0
    style S fill:#1d7a5f,stroke:#2fa882,color:#c8ede5
```
## Dataset

Bộ dữ liệu sử dụng trong dự án này có thể được tải xuống từ:

📊 **[Google Drive Dataset Link](https://drive.google.com/drive/folders/1gz-vo-YugcGh2BVRxsV1SfOOJ-CidT0Z?usp=sharing)**

### Cấu Trúc Dataset
- **Số lượng mẫu**: [X samples]
- **Số lượng feature**: [Y features]
- **Nhãn/Classes**: [Mô tả nhãn]

## Cấu Trúc Dự Án

```
.
├── README.md              # Tài liệu dự án
├── main.ipynb             # Notebook chính - preprocessing, training, evaluation
├── model.pth              # Mô hình đã huấn luyện
└── [Thêm thư mục khác]
```

## Yêu Cầu Môi Trường

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- pandas
- numpy
- scikit-learn
- torch (PyTorch)
- matplotlib / seaborn

## Hướng Dẫn Sử Dụng

1. Tải dataset từ link Google Drive
2. Chạy các cell trong `main.ipynb` theo thứ tự
3. Mô hình sẽ được lưu vào `model.pth`

## Kết Quả

[Thêm kết quả huấn luyện - metrics, accuracy, loss, v.v.]

## Tác Giả



## Ghi Chú

[Thêm bất kỳ ghi chú hoặc hướng phát triển trong tương lai]

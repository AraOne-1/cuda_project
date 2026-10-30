# ใช้ Google Colab GPU ถ้าเครื่องตัวเองไม่มี NVIDIA

โปรเจกต์นี้ใช้ CUDA จริง จึงต้องมี GPU ของ NVIDIA ไม่สามารถรัน kernel CUDA บน Intel UHD / Iris Xe หรือ AMD Radeon ได้โดยตรง

**ทางเลือก:** ใช้ Google Colab Notebook ที่เลือก Hardware accelerator = T4 GPU (หรือ NVIDIA GPU รุ่นอื่นที่ Colab จัดให้) ซึ่งต้องอาศัยสิทธิ์/โควตา GPU ในบัญชีของคุณ ไม่รับประกันว่าจะมี GPU ให้ทุกเวลา

1. เข้า https://colab.research.google.com/ และเปิด Notebook ใหม่
2. ไปที่เมนู Runtime → Change runtime type → Hardware accelerator เลือก **T4 GPU** (ถ้ามีตัวเลือก)
3. อัปโหลดไฟล์ ZIP โครงงานไปที่ Colab ผ่าน Files sidebar (ตัวอย่างชื่อไฟล์ `CUDA_Covertype_Term_Project.zip`)
4. สั่งรันตามเซลล์ด้านล่าง โดยแก้ชื่อไฟล์ ZIP ให้ตรงกับไฟล์ที่อัปโหลด

```python
!nvidia-smi
!nvcc --version
!unzip -q CUDA_Covertype_Term_Project.zip -d cuda_project
%cd cuda_project
!pip -q install -r requirements.txt
!python scripts/prepare_data.py
!make cpu
!make gpu
!python scripts/make_tiny_fixture.py
!python scripts/test_correctness.py ./build/knn_cpu
!./build/knn_cuda --data data/tiny.bin --queries 3 --gpu-baseline
!python scripts/benchmark.py --queries 32 128 512 2048 --gpu-baseline
```

ถ้าเซลล์ `nvidia-smi` บอกว่าไม่พบ GPU ให้ตรวจว่า runtime ใช้ GPU จริง ถ้า `nvcc` ไม่พบ แปลว่าต้องใช้ CUDA Toolkit/Runtime image ที่มี compiler (Colab image แต่ละช่วงอาจต่างกัน)

เมื่อวัดเสร็จให้โหลดไฟล์เหล่านี้ออกจาก Colab:

```python
from google.colab import files
files.download('results/benchmark.csv')
files.download('results/runtime.png')
files.download('results/speedup.png')
```

นำตัวเลขจริงจาก benchmark.csv มาใส่สไลด์ 11 และจดรุ่น GPU, driver, CUDA Toolkit, CPU รุ่น Colab รวมถึงเวลาที่วัด อย่าคัดลอกผลวัดของเครื่องอื่นมาอ้างว่าเป็นของตนเอง

> หมายเหตุ: หาก ZIP มี top-level folder ให้ใช้ `cd` เข้าข้างในอีกชั้นหนึ่ง; ชุด ZIP ที่เตรียมให้นี้ตั้งใจให้ unzip แล้วพบ README.md / Makefile / src/ ในโฟลเดอร์ `cuda_project` โดยตรง

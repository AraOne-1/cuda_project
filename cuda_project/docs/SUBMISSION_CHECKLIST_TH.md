# สิ่งที่ต้องทำก่อนส่งอาจารย์

โจทย์ในไฟล์ต้นฉบับคิดเป็น 20%: Validity 5%, Creativity 5%, Correctness 5%, Presentation 5% และ **ต้องให้เห็นว่า CUDA เร็วกว่า CPU บนข้อมูลใหญ่จริง**

- [x] โจทย์ 1-NN แยกจาก lab vector add / standalone scan / histogram / reduction
- [x] เลือก Dataset จริง UCI Covertype และใส่ Attribution
- [x] เขียน CPU Sequential + การอธิบาย algorithm
- [x] เขียน CUDA พื้นฐานและ CUDA ที่ optimize
- [x] ทำ script download เตรียม dataset และแบ่ง train/query
- [x] ทำ script วัด speedup และสร้างกราฟอัตโนมัติ
- [x] ทำ README, Technical Report, สไลด์ และสคริปต์พูดภาษาไทย
- [x] ตรวจสอบโปรแกรม CPU ผ่านชุดตัวอย่างและ NumPy oracle
- [ ] **ต้องมีเครื่อง NVIDIA ที่ติดตั้ง CUDA Toolkit เพื่อ compile + test CUDA จริง**
- [ ] **ต้องวัด CUDA เร็วกว่า CPU สำหรับ input ขนาดใหญ่ตามข้อกำหนด**
- [ ] นำผล `results/benchmark.csv`, กราฟ, สเปก GPU และ CPU ไปใส่สไลด์ 12
- [ ] ถ่าย/บันทึกผลทดสอบหรือ log สำหรับ demo และซักถาม
- [ ] Push ไฟล์ทั้งหมดขึ้น GitHub ของตนเอง ใส่ link ให้ได้ก่อนกำหนด

**กำหนดส่ง: วันพุธที่ 4 พฤศจิกายน 2026 เวลา 23:59 น.**  
**นำเสนอ: วันพฤหัสบดีที่ 5 พฤศจิกายน 2026 เวลา 13:00–15:00 น.**  
**ใช้เวลานำเสนอแต่ละคนประมาณ 15 นาทีรวม Q&A**

ห้ามส่งสไลด์ที่ตัวเลขยังเป็นเครื่องหมาย '—' เป็นผลจริง หากมี NVIDIA GPU ให้รันคำสั่งที่ระบุใน README แล้วแทนที่ก่อนส่ง

# สคริปต์นำเสนอโครงงาน CUDA Covertype (15 สไลด์)

**แนะนำพูด 11–12 นาที และเผื่อ Q&A ประมาณ 3–4 นาที**

> ก่อนนำเสนอ ต้องคอมไพล์และทดสอบบน NVIDIA GPU จริง แล้วนำผล `results/benchmark.csv` ลงสไลด์ 12 ห้ามอ้างค่า Speedup ที่ยังไม่ได้วัด

### สไลด์ 1 — หัวข้อ (30 วินาที)
โปรเจกต์นี้ใช้ CUDA เร่งการค้นหาข้อมูลป่าไม้ด้วย Exact 1-Nearest Neighbor โดยเปรียบเทียบ CPU ทั้งแบบ naive และ early abandonment กับ CUDA baseline และ optimized GPU

### สไลด์ 2 — ปัญหาและผลลัพธ์ (45 วินาที)
แต่ละ Query ต้องหาข้อมูล Reference ที่ใกล้เคียงที่สุดตาม Features แล้วนำประเภทป่าของ Reference นั้นมาเป็นคำตอบ การเทียบกับข้อมูลหลายแสนรายการซ้ำ ๆ ทำให้เหมาะกับ Parallel Execution

### สไลด์ 3 — Dataset (40 วินาที)
ใช้ UCI Covertype 581,012 Records, 54 Features และ 7 Classes แบ่ง Reference 572,820 และ Query 8,192 ด้วย Seed คงที่ มีการ Standardize 10 Features ด้วยสถิติของ Reference เท่านั้น เพื่อหลีกเลี่ยง Data Leakage

### สไลด์ 4 — สมการ 1-NN (40 วินาที)
ใช้ระยะทาง Euclidean แบบยกกำลังสองจึงไม่ต้อง sqrt เลือก Reference Index ที่ Distance ต่ำสุด แล้วคืน Class ของ Reference ตัวนั้น คำตอบยังเป็น Exact Nearest Neighbor

### สไลด์ 5 — CPU Sequential สองวิธี (60 วินาที)
CPU Naive คำนวณครบทุก Features ทุก Reference ส่วน Early Abandonment จะหยุดทันทีเมื่อ Partial Squared Distance มากกว่าค่าดีที่สุด เพราะระยะทางยกกำลังสองที่เหลือไม่ติดลบ แต่การเช็กเงื่อนไขก็มี Branch Cost ดังนั้นไม่รับรองว่าเร็วกว่าทุกขนาดข้อมูล

### สไลด์ 6 — CUDA Baseline (50 วินาที)
แบ่งแกน Grid Y ตาม Query และ Grid X ตาม Reference Tile แต่ละ Thread คำนวณ Candidates จากนั้นทำ Block Reduction หา Min Pair และ Final Reduction อีก Kernel เพื่อหาคำตอบสุดท้าย

### สไลด์ 7 — Coalescing ด้วย SoA (55 วินาที)
AoS ทำให้ Thread ข้างกันอ่านข้อมูลแบบกระโดด 54 Features แต่ SoA จัด Reference คนละตัวของ Feature เดียวกันให้ติดกัน ช่วยให้ Warp โหลดได้ต่อเนื่อง การแปลง Layout มีค่าใช้จ่ายที่ต้องวัดรวมใน End-to-End

### สไลด์ 8 — Shared Memory, Tiling, Warp Reduction (55 วินาที)
เก็บ Query ไว้ใน Shared Memory เพื่อใช้ซ้ำ ให้ Thread ประมวลผล Reference หลายตัวใน Register เพื่อลด Partial Output และใช้ Warp Shuffle ลดผลลัพธ์โดยไม่ใช้ Global Atomic

### สไลด์ 9 — Work–Step และ APOD (55 วินาที)
งานรวม Worst-case O(QND) ทั้ง CPU และ GPU แต่ GPU กระจาย Candidates พร้อมกัน และมีการลดค่าด้วย Reduction Tree แนวคิด APOD คือ Assess, Parallelize, Optimize และ Deploy ส่วน Step เป็นโมเดลเชิงทฤษฎี ไม่ใช่เวลา GPU ที่วัดจริง

### สไลด์ 10 — Correctness (45 วินาที)
การทำ Min แยก Block แล้วยุบผลรวมยังให้ Minimum เดิม มี Tie-break ด้วย Index ที่น้อยที่สุด ทดสอบ CPU กับ NumPy และต้องทดสอบ GPU กับ CPU เพื่อยืนยันว่าไม่มี Index Mismatch

### สไลด์ 11 — วิธี Benchmark (55 วินาที)
ทดลอง Query Counts 32, 128, 512, 2048 โดยตรึง Reference คงที่ วัด CPU Time, CUDA Event Kernel Time หลัง Warm-up และ End-to-End Time ซึ่งรวม Transpose และ Transfers

### สไลด์ 12 — ผลการทดลอง (60 วินาที; ต้องใส่ผลจริง)
เปิดตาราง CPU ms, GPU Kernel ms, GPU End-to-End ms, Speedup และ Mismatch จาก `results/benchmark.csv` ห้ามกรอกเลขสมมติ ให้ระบุรุ่น GPU/CPU ที่ใช้ และอธิบายว่าเริ่มเร็วกว่าที่ Query Size ไหน หาก GPU ยังช้าต้องวัดใหม่และอธิบายข้อจำกัดก่อนอ้างว่าตรงเกณฑ์

### สไลด์ 13 — Demo (35 วินาที)
สาธิตเตรียม Dataset, Compile `make cpu && make gpu`, ตรวจ Correctness และรัน `scripts/benchmark.py` เพื่อสร้าง CSV และกราฟอัตโนมัติ

### สไลด์ 14 — ข้อจำกัด (45 วินาที)
จำนวน Query น้อยอาจมี Launch และ Transfer Overhead; Reference Tensor ใช้ VRAM; CPU Early Abandonment ลดงานที่เกิดขึ้นจริงได้; Index-based Search หรือ Approximate NN มี Tradeoff อื่น จึงไม่กล่าวว่า Brute-force ดีที่สุดทุกสถานการณ์

### สไลด์ 15 — สรุป (30 วินาที)
โปรเจกต์รวม Dataset จริง, Exact Classification, GPU Parallelism, SoA, Shared Memory และ Reduction พร้อมการทดสอบที่ทำซ้ำได้ ความเร็วต้องพิจารณาจากข้อมูลจริงและ Hardware จริง

## คำถามและคำตอบที่ควรเตรียม

**ทำไม 1-NN?** เป็นงาน Classification ที่ใช้ข้อมูลจริง และการคำนวณ Distance ของแต่ละ Query/Reference แยกเป็น Parallel Candidates ได้

**ทำไมมี CPU สองแบบ?** เพื่อเปรียบเทียบ Straightforward Serial Full Scan กับการลดการคำนวณด้วย Early Abandonment; ไม่บังคับให้ Optimized เร็วกว่าบนทุกชุดข้อมูล

**ทำไมเลือก SoA?** Adjacent Threads ของ Warp จะอ่านตำแหน่ง Reference ต่อเนื่องใน Feature เดียวกัน ช่วย Memory Coalescing

**ทำไมต้องสอง CUDA Kernels?** Block Reduction ภายใน Block แล้ว Final Reduction ข้าม Partial Results เพื่อหลีกเลี่ยงปัญหา Cross-block Synchronization

**Work–Step เท่าไร?** Worst-case Work O(QND); การกระจายงานช่วยลด Critical Path ในโมเดล Ideal แต่ไม่ใช่ GPU Runtime จริง

**อะไรที่ยังไม่ตรวจสอบ?** ต้อง Compile และวัด CUDA GPU จริง, ทำ Cross-validation, บันทึกเลข Speedup/Hardware, และใส่ผลในสไลด์ 12

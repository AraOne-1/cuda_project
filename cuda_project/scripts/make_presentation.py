"""Create an editable 15-slide project defense deck; results intentionally blank.
Run: python scripts/make_presentation.py
Requires: python-pptx (only for regenerating slides).
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

P = Path(__file__).resolve().parents[1]
OUT = P / 'docs' / 'CUDA_Covertype_Presentation.pptx'
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BG = RGBColor(13, 21, 38)
PANEL = RGBColor(24, 37, 57)
PANEL2 = RGBColor(29, 46, 68)
WHITE = RGBColor(241, 247, 253)
MUTED = RGBColor(163, 182, 202)
TEAL = RGBColor(39, 206, 187)
BLUE = RGBColor(78, 157, 249)
AMBER = RGBColor(255, 192, 90)
RED = RGBColor(245, 127, 127)


def rect(sl, x, y, w, h, color=PANEL, radius=True, line=None):
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                              Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background() if line is None else None
    if line is not None:
        shp.line.color.rgb = line
    return shp


def txt(sl, x, y, w, h, text, size=21, color=WHITE, bold=False,
        font='Aptos', align=None, margin=0.06, valign=None):
    shape = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(margin)
    if valign is not None:
        tf.vertical_anchor = valign
    for n, line in enumerate(text.split('\n')):
        p = tf.paragraphs[0] if n == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = font
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = color
        p.space_after = Pt(7)
        if align is not None:
            p.alignment = align
    return shape


def new_slide(title, num, kicker='PROJECT DEFENSE  /  PARALLEL COMPUTING WITH CUDA'):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid(); slide.background.fill.fore_color.rgb = BG
    rect(slide, 0, 0, .12, 7.5, TEAL, radius=False)
    txt(slide, .58, .32, 11, .31, kicker, 11, TEAL, True)
    txt(slide, .56, .82, 12.3, .67, title, 32, WHITE, True)
    rect(slide, .6, 7.05, 12.15, .018, PANEL2, radius=False)
    txt(slide, .62, 7.09, 10.8, .23, 'EXACT 1-NN  •  UCI COVERTYPE  •  54 FEATURES', 10, MUTED)
    txt(slide, 12.05, 7.08, .62, .25, f'{len(prs.slides):02d} / 15', 9, MUTED, align=PP_ALIGN.RIGHT)
    return slide


def label_card(sl, x, y, w, h, label, content, accent=TEAL, fs=19):
    rect(sl, x, y, w, h)
    rect(sl, x + .03, y + .03, .05, h - .06, accent, radius=False)
    txt(sl, x + .23, y + .22, w - .4, .36, label, 13, accent, True)
    txt(sl, x + .23, y + .75, w - .43, h - .86, content, fs, WHITE)


def step(sl, x, y, w, name, detail, accent=TEAL):
    rect(sl, x, y, w, 1.23, PANEL2)
    txt(sl, x+.14, y+.14, w-.3, .3, name, 15, accent, True)
    txt(sl, x+.14, y+.48, w-.3, .66, detail, 15)

# 1 - cover
s = new_slide('Exact 1-NN at Scale', 1, 'CUDA TERM PROJECT  /  INDIVIDUAL PROJECT')
txt(s, .62, 1.78, 11.2, .95, 'GPU-Accelerated Forest Cover Classification', 30, TEAL, True)
txt(s, .64, 2.88, 10.8, .7, '581,012 observations  •  54-dimensional features  •  7 forest classes', 19, MUTED)
rect(s, .62, 4.20, 11.9, 1.44, PANEL)
txt(s, .85, 4.47, 11.4, .96, 'CPU naive/optimized  →  CUDA baseline/optimized\nCoalesced memory  /  shared query cache  /  two-stage argmin', 21)

# 2 - questions
s = new_slide('What are we solving?', 2)
label_card(s,.63,1.85,5.92,3.55,'INPUT','Reference landscape records: 572,820\nHeld-out test queries: 8,192\nEach record contains 54 numeric features.',TEAL,20)
label_card(s,6.75,1.85,5.92,3.55,'OUTPUT','For each query: find the single closest reference in feature space.\nPredict its forest-cover class (1–7).',BLUE,20)
txt(s,.73,5.7,11.85,.62,'Why CUDA?  Every query must compare with many candidate references.',21,AMBER,True)

# 3 - dataset
s = new_slide('A genuine, public large dataset', 3)
for x, big, desc in [(.75,'581,012','LANDSCAPE SAMPLES'),(4.95,'54','INPUT FEATURES'),(9.12,'7','COVER TYPES')]:
    rect(s,x,2.0,3.48,1.52,PANEL)
    txt(s,x+.15,2.23,3.18,.56,big,33,TEAL,True,align=PP_ALIGN.CENTER)
    txt(s,x+.12,2.99,3.27,.32,desc,12,MUTED,True,align=PP_ALIGN.CENTER)
label_card(s,.75,4.02,11.84,1.81,'SPLIT + LEAKAGE CONTROL','10 quantitative features standardized using reference-only statistics. 44 binary features preserved; class label excluded from distance.',BLUE,17)
txt(s,.85,6.19,11.9,.45,'Source: Blackard (1998), UCI Covertype • doi:10.24432/C50K5N • CC BY 4.0',14,MUTED)

# 4 - math
s = new_slide('Exact 1-nearest-neighbor (1-NN)',4)
rect(s,.75,1.87,11.85,1.36,PANEL)
txt(s,1.14,2.20,11.05,.75,'d²(q, xⱼ) = Σ₍f₌₁…₅₄₎ (qᶠ − xⱼᶠ)²',29,TEAL,True,align=PP_ALIGN.CENTER)
label_card(s,.77,3.63,5.73,2.03,'ARGMIN','Find j* with minimum d². Return the class label of reference j*.',BLUE,18)
label_card(s,6.72,3.63,5.73,2.03,'EXACT, NOT APPROXIMATE','No learned model or approximate index; output is a deterministic exhaustive nearest neighbor.',AMBER,18)
txt(s,.86,6.00,11.4,.52,'No square root needed — minimizing d and minimizing d² produce the same winner.',16,MUTED)

# 5 - CPU
s = new_slide('Serial CPU: two fair baselines',5)
rect(s,.7,1.77,7.30,4.62,PANEL)
for i, line in enumerate(['for each query q:',
                         '    best_d2 = +INF; best_idx = -1',
                         '    for each reference x[j]:',
                         '        sum = 0',
                         '        for f in 0..53:',
                         '            sum += (q[f]-x[j][f])²',
                         '            if sum > best_d2: break',
                         '        if sum < best_d2: update best']):
    txt(s,.94,2.04+i*.51,6.90,.41,line,17,TEAL if i in (0,2,4) else WHITE,font='Consolas')
label_card(s,8.24,1.78,4.34,2.0,'CPU NAIVE','Scan all references and all 54 features. Reference for correctness and overhead.',TEAL,18)
label_card(s,8.24,4.00,4.34,2.40,'CPU EARLY ABANDON','Stop when partial squared sum > best. Exact but branching can be slower on small inputs.',BLUE,17)

# 6 - GPU baseline
s = new_slide('CUDA baseline: each thread owns a candidate',6)
step(s,.64,2.05,3.40,'GRID Y','One row of blocks per query',BLUE)
txt(s,4.16,2.4,.40,.4,'→',27,TEAL,True,align=PP_ALIGN.CENTER)
step(s,4.65,2.05,3.68,'GRID X','256 candidate refs per block',TEAL)
txt(s,8.42,2.4,.40,.4,'→',27,TEAL,True,align=PP_ALIGN.CENTER)
step(s,8.90,2.05,3.75,'PARTIAL','Each block writes (min d², index)',AMBER)
rect(s,1.23,4.03,10.88,1.30,PANEL2)
txt(s,1.56,4.29,10.24,.79,'Final CUDA kernel reduces block winners → exact global nearest neighbor.',21,WHITE,True,align=PP_ALIGN.CENTER)
txt(s,.89,5.72,11.75,.59,'Baseline disadvantage: row-major AoS produces wide-stride global reads across a warp.',18,MUTED)

# 7 - memory layout
s = new_slide('Optimization #1: coalesced SoA access',7)
label_card(s,.65,1.82,5.91,3.78,'AoS  •  BASELINE','ref0: f0 f1 f2 f3 … f53\nref1: f0 f1 f2 f3 … f53\nref2: f0 f1 f2 f3 … f53\n\nAdjacent threads access far-apart addresses.',RED,18)
label_card(s,6.72,1.82,5.91,3.78,'SoA  •  OPTIMIZED','f0: ref0 ref1 ref2 ref3 …\nf1: ref0 ref1 ref2 ref3 …\nf2: ref0 ref1 ref2 ref3 …\n\nAdjacent threads load contiguous references.',TEAL,18)
txt(s,.82,6.01,11.85,.58,'Tradeoff: transpose reference matrix once; amortize cost over many queries.',19,AMBER,True)

# 8 - other tuning
s = new_slide('Optimization #2–4: reduce overhead',8)
label_card(s,.65,1.85,3.87,3.95,'SHARED MEMORY','Each block cooperatively caches 54 query features. All threads reuse the same query.',BLUE,20)
label_card(s,4.73,1.85,3.87,3.95,'REGISTER TILING','4 reference candidates per thread; 4× fewer partial block winners than AoS baseline.',TEAL,20)
label_card(s,8.82,1.85,3.87,3.95,'WARP REDUCTION','Use shuffle instructions for min(distance, index); no global atomics.',AMBER,20)
txt(s,.81,6.2,11.77,.38,'Optimizations improve memory efficiency and overhead without changing the exact answer.',16,MUTED)

# Additional slide after 8 - Work–Step + APOD from course lectures
s = new_slide('Work–Step analysis & APOD development',9)
label_card(s,.70,1.80,5.83,2.12,'WORK: W = O(Q × N × D)',
           'Q queries, N references, D features. Exhaustive CPU/CUDA both do O(QND) worst-case arithmetic.',BLUE,16)
label_card(s,6.70,1.80,5.90,2.12,'IDEALIZED PARALLEL STEPS',
           'Per query: O(RD + log B + ceil(T/B) + log B); T = ceil(N/(BR)). Excludes device scheduling.',TEAL,16)
label_card(s,.70,4.13,11.90,1.92,'APOD: ASSESS → PARALLELIZE → OPTIMIZE → DEPLOY',
           'Measure CPU hotspot → map query/reference pairs to GPU → SoA, shared cache, register tiling and reduction → validate + benchmark.',AMBER,17)
txt(s,.90,6.42,11.60,.28,'B=256 threads/block; R=4 references/thread (optimized). Theory ≠ measured GPU milliseconds.',12,MUTED)

# 9 - correctness
s = new_slide('Why the parallel algorithm is correct',9)
label_card(s,.75,1.83,5.77,2.17,'PARTITIONED MINIMUM','min over all references = min of per-block minima.',TEAL,18)
label_card(s,6.72,1.83,5.77,2.17,'CONSISTENT TIES','Compare (distance², row ID) lexicographically.',BLUE,18)
label_card(s,.75,4.21,5.77,2.00,'SERIAL ORACLE TEST','Handmade fixture expects indices [0, 3, 4]. Independent randomized NumPy CPU test passes.',AMBER,17)
label_card(s,6.72,4.21,5.77,2.00,'CUDA VALIDATION','Compare GPU answers with serial answers for every measured query; mismatches flagged.',TEAL,17)

# 10 - method
s = new_slide('How performance is measured',10)
for n,(q,desc) in enumerate([(32,'small'),(128,'medium'),(512,'large'),(2048,'larger')]):
    x=.72+n*3.13
    rect(s,x,1.94,2.90,1.26,PANEL2)
    txt(s,x+.10,2.12,2.72,.48,f'Q = {q}',23,TEAL,True,align=PP_ALIGN.CENTER)
    txt(s,x+.12,2.62,2.68,.33,desc.upper(),11,MUTED,True,align=PP_ALIGN.CENTER)
label_card(s,.75,3.62,5.75,2.51,'SEARCH-ONLY','CPU early-abandon search wall time\nGPU two-kernel elapsed time (CUDA events, after warm-up)',BLUE,17)
label_card(s,6.72,3.62,5.78,2.51,'END-TO-END','GPU AoS→SoA + H2D + one search + D2H\nSpeedup = CPU time / GPU time',TEAL,17)

# 11 - honest blank
s = new_slide('Measured results  •  fill after GPU run',11)
rect(s,.73,1.83,11.93,4.49,PANEL)
txt(s,1.12,2.07,11.18,.52,'Results must come from your own NVIDIA CUDA run',23,AMBER,True,align=PP_ALIGN.CENTER)
columns=[('QUERIES',.95,1.75),('CPU ms',2.75,1.55),('CUDA ms',4.45,1.78),('E2E ms',6.55,1.60),('SPEEDUP',8.52,1.64),('MISMATCHES',10.5,1.87)]
for name,x,w in columns: txt(s,x,3.0,w,.42,name,14,TEAL,True,align=PP_ALIGN.CENTER)
for k,q in enumerate([32,128,512,2048]):
    y=3.52+k*.52
    rect(s,.93,y,11.56,.04,PANEL2,radius=False)
    txt(s,1.15,y+.07,1.17,.33,str(q),14,WHITE,True,align=PP_ALIGN.CENTER)
    for j,x in enumerate([2.75,4.45,6.55,8.52,10.5]):
        txt(s,x,y+.07,1.5,.33,'—',17,MUTED,align=PP_ALIGN.CENTER)
txt(s,1.0,6.41,11.46,.31,'GPU results: NOT MEASURED. Run on actual NVIDIA GPU and replace placeholders before submission.',12,RED)

# 12 - demo
s = new_slide('Live demo: reproducible commands',12)
rect(s,.71,1.81,11.94,4.65,PANEL)
commands=['python -m pip install -r requirements.txt',
          'python scripts/prepare_data.py',
          'make cpu && make gpu',
          './build/knn_cuda --queries 32 --gpu-baseline --cpu-naive',
          'python scripts/benchmark.py --queries 32 128 512 2048 --gpu-baseline']
for i,c in enumerate(commands):
    txt(s,1.05,2.08+i*.70,11.20,.49,c,19,TEAL if i in (3,4) else WHITE,font='Consolas')
txt(s,.94,6.47,11.7,.35,'Windows: compile using nvcc in Developer PowerShell; use build/knn_cuda.exe.',13,MUTED)

# 13 - limitations
s = new_slide('What could limit speedup?',13)
label_card(s,.78,1.96,5.70,1.78,'SMALL WORKLOADS','Kernel launch, transfer and layout-conversion overhead.',AMBER,16)
label_card(s,6.70,1.96,5.70,1.78,'GPU VRAM','Reference tensor and intermediate block winners consume memory.',BLUE,16)
label_card(s,.78,4.00,5.70,1.78,'CPU PRUNING','Early abandon can reduce CPU operations well below the worst case.',TEAL,16)
label_card(s,6.70,4.00,5.70,1.78,'NOT A GLOBAL OPTIMUM','Index-based and approximate searches solve different tradeoffs.',RED,16)

# 14 - conclusion
s = new_slide('Takeaways & references',14)
label_card(s,.75,1.93,11.72,1.43,'PROJECT DELIVERABLE','Reproducible CPU/CUDA implementation + exact agreement checks + hardware-dependent measurements.',TEAL,18)
label_card(s,.75,3.61,11.72,1.28,'MAIN IDEA','Expose one-to-many comparisons; optimize coalescing and reduction, not just thread count.',BLUE,17)
txt(s,.91,5.35,11.5,.48,'UCI Covertype — https://doi.org/10.24432/C50K5N',15,MUTED)
txt(s,.91,5.82,11.5,.48,'NVIDIA — CUDA C++ Programming Guide / Best Practices Guide',15,MUTED)
txt(s,.91,6.40,11.5,.38,'THANK YOU  /  QUESTIONS?',19,TEAL,True)

OUT.parent.mkdir(parents=True,exist_ok=True)
prs.save(OUT)
print('Wrote', OUT, 'slides=', len(prs.slides))

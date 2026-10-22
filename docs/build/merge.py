import fitz
body = fitz.open("body.pdf"); n = len(body)
titles = ["Document control","Purpose and design principle","System architecture","Visual kinematics","Bioacoustic analysis",
          "Neuro-symbolic fusion","The monochrome HUD","Models and training results","Running the system",
          "Interfaces and file layout","Verification","Field test with Rokid glasses footage","Limitations and open issues","Next steps"]
# find heading page (first page after TOC whose text contains the title as a heading line)
pages = {}
for t in titles:
    for i in range(1, n):
        if any(l.strip() == t for l in body[i].get_text().splitlines()):
            pages[t] = i + 1; break
toc = body[0]
W = toc.rect.width
for t, p in pages.items():
    r = toc.search_for(t)[0]
    s = str(p); w = fitz.get_text_length(s, "helv", 11)
    toc.insert_text((W - 22 * 72 / 25.4 - w, r.y1 - 2), s, fontname="helv", fontsize=11, color=(0.42, 0.46, 0.53))
for i, pg in enumerate(body):
    W, H = pg.rect.width, pg.rect.height; m = 22 * 72 / 25.4
    pg.insert_text((m, 30), "SYMBOLIC SYNAPSE", fontname="hebo", fontsize=6.5, color=(0.19, 0.37, 0.65))
    pg.insert_text((m + fitz.get_text_length("SYMBOLIC SYNAPSE", "hebo", 6.5) + 4, 30), "·  DogSense HUD", fontname="helv", fontsize=6.5, color=(0.45, 0.5, 0.58))
    rt = "Technical documentation of the canine behaviour telemetry prototype"
    pg.insert_text((W - m - fitz.get_text_length(rt, "helv", 6.5), 30), rt, fontname="helv", fontsize=6.5, color=(0.45, 0.5, 0.58))
    pg.insert_text((m, H - 26), "Official document · version 1.4 · 22 October 2026", fontname="helv", fontsize=6.5, color=(0.45, 0.5, 0.58))
    pn = f"Page {i + 1} of {n}"
    pg.insert_text((W - m - fitz.get_text_length(pn, "helv", 6.5), H - 26), pn, fontname="helv", fontsize=6.5, color=(0.45, 0.5, 0.58))
out = fitz.open("cover.pdf"); out.select([0]); out.insert_pdf(body)
out.set_metadata({"title": "DogSense HUD - Technical documentation", "author": "Symbolic Synapse"})
import sys
dst = sys.argv[1] if len(sys.argv) > 1 else "../DogSense_HUD_Technical_Documentation.pdf"
out.save(dst, garbage=3, deflate=True)
print(n, pages)

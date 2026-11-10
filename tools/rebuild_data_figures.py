"""Extract existing plotted samples losslessly and redraw legible vector labels.

This preserves the submitted visual sample content; it does not run a model.
The manifest records the PDF and image sources used for the first extraction.
"""
from pathlib import Path
import hashlib
import json
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'typst'/'figures'/'data'
SPECS={
 'fig_exp_pixel_forward_reverse':12,
 'fig_exp_ldm_latent_and_noise':3,
 'fig_exp_ldm_cross_attn_conditioning':4,
}

def prepare():
 DATA.mkdir(parents=True,exist_ok=True)
 manifest_path=DATA/'provenance.json'
 if manifest_path.exists():
  manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
  for record in manifest['panels']:
   assert hashlib.sha256((DATA/record['file']).read_bytes()).hexdigest()==record['png_sha256']
  return
 manifest={'description':'Lossless extraction of plotted sample images; labels redrawn separately. No model inference.','panels':[]}
 for name,count in SPECS.items():
  source=ROOT/'figures'/f'{name}.pdf'
  doc=pymupdf.open(source)
  infos=doc[0].get_image_info(xrefs=True)
  assert len(infos)==count,(name,len(infos))
  masks={im[0]:im[1] for im in doc[0].get_images(full=True)}
  infos.sort(key=lambda im:(round(im['bbox'][1]),im['bbox'][0]))
  for index,info in enumerate(infos):
   pix=pymupdf.Pixmap(doc,info['xref'])
   if masks[info['xref']]:
    pix=pymupdf.Pixmap(pix,pymupdf.Pixmap(doc,masks[info['xref']]))
   filename=f'{name}-{index:02}.png'
   pix.save(DATA/filename)
   manifest['panels'].append({'file':filename,'source_pdf':str(source.relative_to(ROOT)),
    'source_pdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'source_image_xref':info['xref'],'png_sha256':hashlib.sha256((DATA/filename).read_bytes()).hexdigest()})
 manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__':
 prepare()

# merge_to_coco.py
from pathlib import Path
import json, itertools

def poly_area(poly):
    # shoelace
    x = [p[0] for p in poly]; y = [p[1] for p in poly]
    return 0.5*abs(sum(x[i]*y[(i+1)%len(poly)] - x[(i+1)%len(poly)]*y[i] for i in range(len(poly))))

def bbox_from_poly(poly):
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    x0, y0 = min(xs), min(ys); x1, y1 = max(xs), max(ys)
    return [float(x0), float(y0), float(x1-x0), float(y1-y0)]

def main(images_dir="images", anns_dir="anns", out_json="coco.json"):
    anns_dir = Path(anns_dir)
    items = sorted(anns_dir.glob("*.json"))
    images = []; annotations = []
    cat = {"id":1, "name":"leaf", "supercategory":"plant"}
    ann_id = 1; img_id = 1
    for j in items:
        data = json.loads(j.read_text())
        w,h = data["image_size"]
        images.append({"id": img_id, "file_name": data["image"], "width": w, "height": h})
        for poly in data["polygons"]:
            if len(poly) < 3: continue
            seg = list(itertools.chain.from_iterable(poly))
            area = poly_area(poly)
            bbox = bbox_from_poly(poly)
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "segmentation": [seg], "iscrowd": 0, "area": float(area), "bbox": bbox
            })
            ann_id += 1
        img_id += 1
    coco = {"images": images, "annotations": annotations, "categories": [cat]}
    Path(out_json).write_text(json.dumps(coco, indent=2))
    print(f"Wrote {out_json} with {len(images)} images and {len(annotations)} instances.")

if __name__ == "__main__":
    main()

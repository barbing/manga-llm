
import cv2
import numpy as np

def detect_mser_boxes(img_bgr, min_area=200, max_area_ratio=0.2, delta=5, max_variation=0.2):
    r"""Return list of (x1,y1,x2,y2) candidate text regions using MSER.

    Works well for high-contrast manga text.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Slight blur to stabilize MSER
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    mser = cv2.MSER_create(_delta=delta, _max_variation=max_variation)
    regions, _ = mser.detectRegions(gray)

    boxes = []
    max_area = int(max_area_ratio * w * h)
    for p in regions:
        x,y,w0,h0 = cv2.boundingRect(p.reshape(-1,1,2))
        area = w0*h0
        if area < min_area or area > max_area:
            continue
        # expand a bit
        pad = int(0.02 * (w0+h0))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w-1, x + w0 + pad)
        y2 = min(h-1, y + h0 + pad)
        boxes.append((x1,y1,x2,y2))

    # Merge overlapping boxes via NMS-like union
    boxes = merge_boxes(boxes, iou_thresh=0.2)
    return boxes

def merge_boxes(boxes, iou_thresh=0.2):
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.int32)
    picked = []
    used = np.zeros(len(boxes_np), dtype=bool)

    def iou(a,b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        inter_x1 = max(ax1,bx1); inter_y1 = max(ay1,by1)
        inter_x2 = min(ax2,bx2); inter_y2 = min(ay2,by2)
        iw = max(0, inter_x2 - inter_x1); ih = max(0, inter_y2 - inter_y1)
        inter = iw*ih
        if inter == 0: return 0.0
        area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter + 1e-6)

    for i in range(len(boxes_np)):
        if used[i]: continue
        cur = boxes_np[i].tolist()
        used[i] = True
        # try to merge with others
        merged = True
        while merged:
            merged = False
            for j in range(i+1, len(boxes_np)):
                if used[j]: continue
                if iou(cur, boxes_np[j]) > iou_thresh:
                    # union
                    x1 = min(cur[0], boxes_np[j][0])
                    y1 = min(cur[1], boxes_np[j][1])
                    x2 = max(cur[2], boxes_np[j][2])
                    y2 = max(cur[3], boxes_np[j][3])
                    cur = [x1,y1,x2,y2]
                    used[j] = True
                    merged = True
        picked.append(tuple(cur))
    return picked

from fastapi import FastAPI, File, UploadFile,WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2
import json
import base64
import numpy as np
import torch
from torchvision.transforms import functional as F
from transformers.modeling_outputs import BaseModelOutput
import math
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.nn.functional as Fn


# Your model setup code from before remains the same...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Pose adapter model
class GraphAttentionPoseAdapter2(torch.nn.Module):
    def __init__(self, joint_dim, num_joints, output_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        self.input_proj = torch.nn.Linear(joint_dim, hidden_dim)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, num_joints, hidden_dim))
        self.input_norm = torch.nn.LayerNorm(hidden_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = torch.nn.Linear(hidden_dim * num_joints, output_dim)
        self.output_norm = torch.nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 16, 2)
        x = self.input_proj(x)
        x = self.input_norm(x + self.pos_embedding)
        x, _ = self.attn(x, x, x)
        x = x.flatten(start_dim=1)
        return self.output_norm(self.output_proj(x))


# Load keypoint detector
keypoint_model = keypointrcnn_resnet50_fpn(pretrained=True).eval().to(device)

# Load tokenizer & T5
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)

# Load pose adapter & prefix projector
pose_adapter = GraphAttentionPoseAdapter2(joint_dim=2, num_joints=16, output_dim=768).to(device)
pose_to_prefix = torch.nn.Linear(768, 512 * 4).to(device)

checkpoint = torch.load("/Users/eishahemchand/postureCorrection/mpiipose_feedback_model.pt", map_location=device)
pose_adapter.load_state_dict(checkpoint["pose_adapter"])
# Linear projection to prefix
pose_to_prefix = torch.nn.Linear(768, 512 * 4).to(device)
# pose_to_prefix.load_state_dict(checkpoint["pose_to_prefix"])
t5_model.load_state_dict(checkpoint["t5_model"])

pose_adapter.eval()
# pose_to_prefix.eval()
t5_model.eval()

# ====== Util Functions ======
def convert_coco_to_mp11(kp):
    kp = np.array(kp).reshape(-1, 2)
    mp11 = np.zeros((16, 2), dtype=np.float32)
    mp11[0]  = kp[16]; mp11[1]  = kp[14]; mp11[2]  = kp[12]; mp11[3]  = kp[11]
    mp11[4]  = kp[13]; mp11[5]  = kp[15]; mp11[6]  = (kp[11] + kp[12]) / 2
    mp11[7]  = (kp[5] + kp[6]) / 2; mp11[8]  = kp[0]; mp11[9]  = (kp[1] + kp[2]) / 2
    mp11[10] = kp[10]; mp11[11] = kp[8]; mp11[12] = kp[6]; mp11[13] = kp[5]
    mp11[14] = kp[7]; mp11[15] = kp[9]
    return mp11

def angle_between(p1, p2, p3):
    a, b, c = torch.tensor(p1), torch.tensor(p2), torch.tensor(p3)
    ba = a - b
    bc = c - b
    cos_angle = Fn.cosine_similarity(ba.unsqueeze(0), bc.unsqueeze(0), dim=1).clamp(-1.0, 1.0)
    return torch.acos(cos_angle).item() * (180 / math.pi)

def infer_posture_label_coco_old(keypoints):
    kp = {
        i: keypoints[i]
        for i in range(len(keypoints))
        if keypoints[i] is not None and not (keypoints[i][0] == 0 and keypoints[i][1] == 0)
    }

    labels = []

    if 5 in kp and 7 in kp and (kp[7][1] - kp[5][1] < 10):
        labels.append("slouching shoulders")
    if 6 in kp and 8 in kp and (kp[8][1] - kp[6][1] < 10):
        labels.append("drooping right shoulder")
    if 5 in kp and 6 in kp and (abs(kp[5][1] - kp[6][1]) > 20):
        labels.append("shoulder imbalance")
    if 5 in kp and 11 in kp and (abs(kp[5][1] - kp[11][1]) < 30):
        labels.append("arched back")
    if 6 in kp and 12 in kp and (abs(kp[6][1] - kp[12][1]) < 30):
        labels.append("arched back")
    if 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        spine_angle = angle_between(
            kp[5], [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2], kp[11]
        )
        if spine_angle < 150:
            labels.append("rounded upper back")
    if 11 in kp and 12 in kp and abs(kp[11][1] - kp[12][1]) > 20:
        labels.append("hip tilt")
    if 13 in kp and 15 in kp and abs(kp[13][0] - kp[15][0]) > 40:
        labels.append("bowed left leg")
    if 14 in kp and 16 in kp and abs(kp[14][0] - kp[16][0]) > 40:
        labels.append("bowed right leg")
    if 11 in kp and 13 in kp and kp[13][1] < kp[11][1]:
        labels.append("hyperextended left knee")
    if 12 in kp and 14 in kp and kp[14][1] < kp[12][1]:
        labels.append("hyperextended right knee")
    if 0 in kp and 1 in kp and 2 in kp and abs(kp[1][1] - kp[2][1]) > 10:
        labels.append("head tilt")
    if 0 in kp and 5 in kp and 6 in kp and 11 in kp and 12 in kp:
        head_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
        torso_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
        neck_angle = angle_between(kp[0], head_mid, torso_mid)
        if neck_angle < 160:
            labels.append("forward head posture")

    return labels if labels else ["good posture"]


def infer_posture_label_coco(keypoints_with_scores, img_height=480, confidence_thresh=0.5, edge_buffer=15):
    """
    keypoints_with_scores: np.array of shape [17, 3] (x, y, score) from keypoint RCNN
    """
    keypoints = keypoints_with_scores[:, :2]
    scores = keypoints_with_scores[:, 2]

    def is_valid(idx):
        # keypoint must not be near (0, 0), must be within image bounds, and above confidence
        if scores[idx] < confidence_thresh:
            return False
        x, y = keypoints[idx]
        if np.allclose([x, y], [0, 0], atol=2):
            return False
        if y >= img_height - edge_buffer:
            return False
        return True

    def has_valid_keys(*ids):
        return all(is_valid(i) for i in ids)

    kp = {i: keypoints[i] for i in range(len(keypoints)) if is_valid(i)}

    labels = []

    if has_valid_keys(5, 7) and (abs(kp[7][1] - kp[5][1]) < 10):
        labels.append("slouching shoulders")
    if has_valid_keys(6, 8) and (abs(kp[8][1] - kp[6][1]) < 10):
        labels.append("drooping right shoulder")
    if has_valid_keys(5, 6) and (abs(kp[5][1] - kp[6][1]) > 20):
        labels.append("shoulder imbalance")
    if has_valid_keys(5, 11) and (abs(kp[5][1] - kp[11][1]) < 30):
        labels.append("arched back")
    if has_valid_keys(6, 12) and (abs(kp[6][1] - kp[12][1]) < 30):
        labels.append("arched back")
    if has_valid_keys(5, 6, 11):
        spine_angle = angle_between(
            kp[5], [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2], kp[11]
        )
        if spine_angle < 150:
            labels.append("rounded upper back")
    if has_valid_keys(11, 12) and abs(kp[11][1] - kp[12][1]) > 20:
        labels.append("hip tilt")
    if has_valid_keys(13, 15) and abs(kp[13][0] - kp[15][0]) > 40:
        labels.append("bowed left leg")
    if has_valid_keys(14, 16) and abs(kp[14][0] - kp[16][0]) > 40:
        labels.append("bowed right leg")
    if has_valid_keys(11, 13) and kp[13][1] < kp[11][1]:
        labels.append("hyperextended left knee")
    if has_valid_keys(12, 14) and kp[14][1] < kp[12][1]:
        labels.append("hyperextended right knee")
    if has_valid_keys(0, 1, 2) and abs(kp[1][1] - kp[2][1]) > 10:
        labels.append("head tilt")
    if has_valid_keys(0, 5, 6, 11, 12):
        head_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
        torso_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
        neck_angle = angle_between(kp[0], head_mid, torso_mid)
        if neck_angle < 160:
            labels.append("forward head posture")

    return labels if labels else ["good posture"]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws/posture")
async def websocket_posture(ws: WebSocket):
    await ws.accept()
    while True:
        try:
            data = json.loads(await ws.receive_text())
            frame_data = data["frame"]
            role = data.get("role", "dancer")

            decoded_bytes = base64.b64decode(frame_data.split(",")[1])
            nparr = np.frombuffer(decoded_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_tensor = F.to_tensor(img).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            with torch.no_grad():
                output = keypoint_model([img_tensor])[0]

            if len(output['keypoints']) == 0:
                await ws.send_json({"feedback": [{"label": "none", "message": "No person detected.", "severity": "none"}]})
                continue

            keypoints = output['keypoints'][0][:, :2].cpu().numpy()
            print(keypoints)
            mp11 = convert_coco_to_mp11(keypoints)
            label_list = infer_posture_label_coco_old(keypoints)

            pose_tensor = torch.tensor(mp11, dtype=torch.float32).unsqueeze(0).to(img_tensor.device)
            pose_emb = pose_adapter(pose_tensor)
            prefix_emb = pose_to_prefix(pose_emb).view(1, 4, 512)

            feedback_list = []
            for label in label_list:
                # Mock severity assignment
                severity = "moderate" if "shoulder" in label else "mild"
                prompt = f"Observed this posture flaw: {label}. Feedback:"
                tokenized = tokenizer(prompt, return_tensors="pt").to(device)
                token_emb = t5_model.encoder.embed_tokens(tokenized.input_ids)
                encoder_input = torch.cat([prefix_emb, token_emb], dim=1)
                encoder_outputs = BaseModelOutput(last_hidden_state=encoder_input)
                # prompt = f"As a {role}, observed flaw: {label}. Suggest feedback:"
                # tokenized = tokenizer(prompt, return_tensors="pt").to(img_tensor.device)
                # token_emb = t5_model.encoder.embed_tokens(tokenized.input_ids)
                # encoder_input = torch.cat([prefix_emb, token_emb], dim=1)
                # encoder_outputs = BaseModelOutput(last_hidden_state=encoder_input)
                gen_ids = t5_model.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=64,
                    num_beams=5,
                    early_stopping=True
                )
                feedback = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                feedback_list.append({
                    "label": label,
                    "message": feedback,
                    "severity": severity
                })

            await ws.send_json({"feedback": feedback_list})

        except Exception as e:
            print(f"WebSocket error: {e}")
            await ws.close()
            break

# @app.post("/posture")
# async def posture_feedback(frame: UploadFile = File(...)):
#     contents = await frame.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Convert to tensor
#     img_tensor = F.to_tensor(img).to(device)

#     with torch.no_grad():
#         output = keypoint_model([img_tensor])[0]

#     if len(output['keypoints']) == 0:
#         return {"feedback": [{"label": "none", "message": "No person detected."}]}

#     keypoints = output['keypoints'][0][:, :2].cpu().numpy()
#     mp11 = convert_coco_to_mp11(keypoints)
#     labels = infer_posture_label_coco(keypoints)

#     pose_tensor = torch.tensor(mp11, dtype=torch.float32).unsqueeze(0).to(device)
#     feedback_list = []

#     with torch.no_grad():
#         pose_emb = pose_adapter(pose_tensor)
#         prefix_emb = pose_to_prefix(pose_emb).view(1, 4, 512)

#         for label in labels:
#             prompt = f"Observed this posture flaw: {label}. Feedback:"
#             tokenized = tokenizer(prompt, return_tensors="pt").to(device)
#             token_emb = t5_model.encoder.embed_tokens(tokenized.input_ids)
#             encoder_input = torch.cat([prefix_emb, token_emb], dim=1)
#             encoder_outputs = BaseModelOutput(last_hidden_state=encoder_input)
#             gen_ids = t5_model.generate(
#                 encoder_outputs=encoder_outputs,
#                 max_length=64,
#                 num_beams=5,
#                 early_stopping=True
#             )
#             feedback = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
#             feedback_list.append({"label": label, "message": feedback})

#     return {"feedback": feedback_list}

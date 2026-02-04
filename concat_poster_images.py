from PIL import Image
import os

# 按照你要求的顺序填写图片路径
image_paths = [
    "/home/zechuan/policyReader/policy_outputs/posters/1770174849_cover_tail_0_cover.png",
    "/home/zechuan/policyReader/policy_outputs/posters/1770174023_what_0.png",
    "/home/zechuan/policyReader/policy_outputs/posters/1770174066_threshold_0.png",
    "/home/zechuan/policyReader/policy_outputs/posters/1770174133_compliance_0.png",
    "/home/zechuan/policyReader/policy_outputs/posters/1770174849_cover_tail_0_tail.png",
]

# 输出路径可以根据需要改
output_path = "/home/zechuan/policyReader/policy_outputs/posters/merged_long_poster.png"

# 加载图片
images = []
for p in image_paths:
    try:
        img = Image.open(p).convert("RGB")
        images.append(img)
    except Exception as e:
        print(f"打开图片失败: {p} ({e})")

if not images:
    raise RuntimeError("没有成功打开任何图片")

# 统一宽度，参考 poster_pipeline.concat_poster_images 的做法：
# 取所有图片 width 的最小值，其他按比例缩放到这个宽度
target_width = min(img.width for img in images)

resized_images = []
total_height = 0
for img in images:
    if img.width == target_width:
        resized = img
    else:
        new_height = int(img.height * target_width / img.width)
        resized = img.resize((target_width, new_height), resample=Image.LANCZOS)
    resized_images.append(resized)
    total_height += resized.height

# 创建白底长图画布（竖向拼接）
long_img = Image.new("RGB", (target_width, total_height), color="white")

y_offset = 0
for img in resized_images:
    long_img.paste(img, (0, y_offset))
    y_offset += img.height

os.makedirs(os.path.dirname(output_path), exist_ok=True)
long_img.save(output_path)
print(f"拼接完成，已保存至: {output_path}")
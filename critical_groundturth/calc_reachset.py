from nuscenes.nuscenes import NuScenes


nusc = NuScenes(version='v1.0-mini', dataroot=r'E:\Linux-ubuntu\datasets\nuscenes_mini', verbose=False)

scene = nusc.scene[0]
category = "human.pedestrian"

pedestrian_instances = {}

sample_token = scene['first_sample_token']

while sample_token:
    sample = nusc.get('sample', sample_token)

    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)

        if ann['category_name'].startswith(category + '.'):
            instance_token = ann['instance_token']

            if instance_token not in pedestrian_instances:
                pedestrian_instances[instance_token] = {
                    'category': ann['category_name'],
                    'annotations': []
                }

            pedestrian_instances[instance_token]['annotations'].append(
                ann_token
            )

    sample_token = sample['next']

# 打印所有Pedestrian instances
for idx, (token, info) in enumerate(pedestrian_instances.items()):
    print(
        f"{idx}: {token} | "
        f"{info['category']} | "
        f"{len(info['annotations'])} samples"
    )





























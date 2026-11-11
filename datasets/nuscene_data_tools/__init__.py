'''
    本部分代码为对nuscene数据集的解码

    nuscene_2dT3d: 复制于官方代码，将3d anno转换为2d坐标(x1, y1, x2, y2)，文件会存储到nuscene数据集中存放各类json文件的文件夹中。

    nuscenes_scene_anno：将anno按场景解码为统一的json文件，每个场景都存放为单独的json
    涵盖所有不同距离，不同occlusion程度的行人

'''
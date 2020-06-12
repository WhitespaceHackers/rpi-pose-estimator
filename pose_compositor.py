import json
import matplotlib.pyplot as plt
import os

json_dir = os.path.abspath('pose_estimates')
image_dir = os.path.abspath('images')
composite_dir = os.path.abspath('pose_composites')

threshold = 0.25

for fn in sorted(filter(lambda x: x.endswith('.json'), os.listdir(json_dir))):
    with open(os.path.join(json_dir, fn), 'r') as f:
        pose_data = json.load(f)
    print(fn)
    image = plt.imread(os.path.join(image_dir, fn.replace('json','jpg')))
    plt.imshow(image)
    for point in pose_data['keypoints']:
        if point['score'] > threshold:
            plt.plot(point['position']['x'], point['position']['y'], 'ro')
    plt.title('score: ' + str(pose_data['score']))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(composite_dir, fn.replace('json','jpg')), dpi=300)
    plt.clf()

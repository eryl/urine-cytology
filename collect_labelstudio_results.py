import argparse
from pathlib import Path
import json
import urllib
import urllib.parse

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('labelstudio_target_dir', type=Path)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('output_file', type=Path)
    args = parser.parse_args()
    images = {img_path.name: img_path for img_path in args.image_dir.glob('*.png')}
    annotation_files = list(args.labelstudio_target_dir.iterdir())
    
    annotation_records = []
    for annotation_file in annotation_files:
        with open(annotation_file) as fp:
            annotation = json.load(fp)
            image_url = annotation['task']['data']['image']
            parse_result = urllib.parse.urlparse(image_url)
            image_path = Path(urllib.parse.parse_qs(parse_result.query)['d'][0])
            image_name = image_path.name
            if image_name not in images:
                print(f"image {image_name} not found")
            else:
                annotation_result = annotation['result']
                ncratio = None
                quality = None
                isvalid = None
                hyperchromasia = False
                coarse_chromatin = False
                irregular_chromatin = False
                is_group = False
                reactive_atypia = False
                inflammatory = False
                not_a_cell = False
                no_cytoplasma = False
                for annotation_dict in annotation_result:
                    annotation_type = annotation_dict['from_name'] 
                    annotation_value = set(annotation_dict['value']['choices'])
                    if annotation_type == 'ncratio':
                        ncratio, = annotation_value
                    elif annotation_type == 'quality':
                        quality, = annotation_value
                    elif annotation_type == 'isvalid':
                        if 'Reactive atypia' in annotation_value:
                            annotation_value.remove('Reactive atypia')
                            reactive_atypia = True
                        if 'Cell group' in annotation_value:
                            annotation_value.remove('Cell group')
                            is_group = True
                        if 'Inflammatory' in annotation_value:
                            annotation_value.remove('Inflammatory')
                            inflammatory = True
                        if 'Not a cell' in annotation_value:
                            annotation_value.remove('Not a cell')
                            not_a_cell = True
                        if 'No cytoplasma' in annotation_value:
                            annotation_value.remove('No cytoplasma')
                            no_cytoplasma = True
                        if len(annotation_value) > 0:
                            raise NotImplementedError(f"Annotation value has not been handled {annotation_value}")
                        
                    elif annotation_type == 'chromatin-structure':
                        if 'Hyperchromasia' in annotation_value:
                            annotation_value.remove('Hyperchromasia')
                            hyperchromasia = True
                        if 'Coarse chromatin' in annotation_value:
                            annotation_value.remove('Coarse chromatin')
                            coarse_chromatin = True
                        if 'Irregular Chromatin Rim' in annotation_value:
                            annotation_value.remove('Irregular Chromatin Rim')
                            irregular_chromatin = True
                        if len(annotation_value) > 0:
                            raise NotImplementedError(f"Annotation value has not been handled {annotation_value}")
                    else:
                        raise NotImplementedError(f"Annotation type {annotation_type} has not been implemented")
                    print(annotation_dict)
                
                record = {'image': image_name, 
                          'nc_ratio': ncratio, 
                          'quality': quality, 
                          'validity': isvalid,
                          'hyperchromasia': hyperchromasia,
                            'coarse_chromatin': coarse_chromatin,
                            'irregular_chromatin': irregular_chromatin,
                            'is_group': is_group,
                            'reactive_atypia': reactive_atypia,
                            'inflammatory': inflammatory,
                            'not_a_cell': not_a_cell,
                            'no_cytoplasma': no_cytoplasma}
                annotation_records.append(record)
    
    df = pd.DataFrame.from_dict(annotation_records)
    df.to_csv(args.output_file, index=False)
    
if __name__ == '__main__':
    main()
    
    
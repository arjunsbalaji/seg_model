#!/usr/bin/env python3

from exp.nb_detectron import *
import mlflow
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
'''
This python file takes 3 CLI arguments checkpoint_name, thresh, test. All are ints. Defaults listed below.
'''

try:
    checkpoint_name = sys.argv[1]
    thresh = float(sys.argv[2])
    size = str(sys.argv[3])
    test = int(sys.argv[4])==True
except:
    checkpoint_name = 'dummy'
    thresh = 0.7
    size = 0
    

if test: runtype = 'test'
else: runtype = 'valid'

if size == 0: anno_file_name = 'medium_set_annotations.json'
else:

    
projectname = 'OCT'
data_path = Path('/workspace/oct_ca_seg/COCOdata/')
valid_path = data_path/'valid'
test_path = data_path/'test'

validCOCO = COCO(valid_path/'images/annotations.json')
testCOCO = COCO(test_path/'images/annotations.json')

for d in [valid_path, test_path]:
    DatasetCatalog.register(projectname + d.name,
                            lambda d=d: load_coco_json(d/('images/annotations.json'), d/'images', dataset_name=d.name))  #get_dicts(d.name))#
    MetadataCatalog.get(projectname+ d.name).set(stuff_classes=["lumen"])
    
    
valid_metadata = MetadataCatalog.get(projectname+'valid')
valid_metadata.stuff_classes = ['lumen']
valid_metadata.thing_classes = ['lumen']
valid_metadata.json_file = str(valid_path/('images/'+anno_file_name))

test_metadata = MetadataCatalog.get(projectname+'test')
test_metadata.stuff_classes = ['lumen']
test_metadata.thing_classes = ['lumen']
test_metadata.json_file = str(test_path/('images/'+anno_file_name))

cfg = get_cfg()
cfg.merge_from_file('/workspace/oct_ca_seg/runsaves/'+ checkpoint_name + '/01_OCTPawsey_model_mask_rcnn_R_50_FPN_3x.yaml') #configD2.yaml')

cfg.DATASETS.TEST = (projectname+runtype,)


print(cfg.DATASETS.TEST)
cfg.OUTPUT_DIR = ('/workspace/oct_ca_seg/runsaves/'+checkpoint_name)


tracking_uri = 'file:/workspace/oct_ca_seg/runsaves/mlruns'
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    
    mlflow.log_param('checkpoint_name',checkpoint_name)
    mlflow.log_param('thresh',thresh)
    mlflow.log_param('test',test)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)


    coco_ev = COCOEvaluator(projectname+runtype, cfg, False, output_dir=cfg.OUTPUT_DIR)
    OCT_ev = OCT_Evaluator(validCOCO)

    evaluators = DatasetEvaluators([coco_ev, OCT_ev])
    val_loader = build_detection_test_loader(cfg, projectname+runtype)
    results = inference_on_dataset(predictor.model, val_loader, evaluators)

    #client = mlflow.tracking.MlflowClient()
    runname = mlflow.active_run().info.run_id
    save_results(results, str(Path(cfg.OUTPUT_DIR)/(runname+'_results.json')))
    
    stats = {'u_dice': np.mean(list(results['dices'].values())),
             'u_spec': np.mean(list(results['specs'].values())),
             'u_sens': np.mean(list(results['sens'].values())),
             'u_acc': np.mean(list(results['accs'].values()))}
             
    mlflow.log_metrics(stats)
    mlflow.log_artifact(cfg.OUTPUT_DIR+'/results.json')
    #mlflow.pytorch.log_model(predictor.model, '')
    #mlflow.pytorch.save_model(predictor.model)

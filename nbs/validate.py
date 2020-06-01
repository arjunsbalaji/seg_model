#!/usr/bin/env python3

from exp.nb_detectron import *
import mlflow
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

checkpoint_name = sys.argv[1]
thresh = sys.argv[2]
test = sys.argv[3]==True

if test: runtype = 'test'
else: runtype = 'valid'

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
test_metadata = MetadataCatalog.get(projectname+'test')
test_metadata.stuff_classes = ['lumen']
test_metadata.thing_classes = ['lumen']

cfg = get_cfg()
cfg.merge_from_file('/workspace/oct_ca_seg/runsaves/'+ checkpoint_name + '/configD2.yaml')
cfg.OUTPUT_DIR = ('/workspace/oct_ca_seg/runsaves/'+checkpoint_name)

with mlflow.start_run():
    
    
    
    mlflow.log_param('checkpoint_name',checkpoint_name)
    mlflow.log_param('thresh',thresh)
    mlflow.log_param('test',test)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)


    coco_ev = COCOEvaluator(projectname+'_'+runtype, cfg, False, output_dir=cfg.OUTPUT_DIR)
    OCT_ev = OCT_Evaluator(validCOCO)

    evaluators = DatasetEvaluators([coco_ev, OCT_ev])
    val_loader = build_detection_test_loader(cfg, projectname+'_'+runtype)
    results = inference_on_dataset(predictor.model, val_loader, evaluators)

    #client = mlflow.tracking.MlflowClient()
    runname = mlflow.active_run().info.run_id
    save_results(results, Path(cfg.OUTPUT_DIR)/(runname+'_results.json')
    
    stats = {'u_dice': np.mean(list(results['dices'].values())),
             'u_spec': np.mean(list(results['specs'].values())),
             'u_sens': np.mean(list(results['sens'].values())),
             'u_acc': np.mean(list(results['accs'].values()))}
             
    mlflow.log_metrics(stats)
    mlflow.pytorch.log_model(predictor.model, registered_model_name='first mlflow try')
    mlflow.pytorch.save_model(predictor.model)

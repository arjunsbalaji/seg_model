#!/usr/bin/env python3

from exp.nb_detectron import *
import mlflow
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

argv
data_path = Path('/workspace/oct_ca_seg/COCOdata/')
train_path = data_path/'train'
valid_path = data_path/'valid'
test_path = data_path/'test'

trainCOCO = COCO(train_path/'images/annotations.json')
validCOCO = COCO(valid_path/'images/annotations.json')
testCOCO = COCO(test_path/'images/annotations.json')

for d in [train_path, valid_path, test_path]:
    DatasetCatalog.register(projectname + d.name,
                            lambda d=d: load_coco_json(d/('images/annotations.json'), d/'images', dataset_name=d.name))  #get_dicts(d.name))#
    MetadataCatalog.get(projectname+ d.name).set(stuff_classes=["lumen"])
    
train_metadata = MetadataCatalog.get(projectname+'train')
train_metadata.stuff_classes = ['lumen']
train_metadata.thing_classes = ['lumen']
valid_metadata = MetadataCatalog.get(projectname+'valid')
valid_metadata.stuff_classes = ['lumen']
valid_metadata.thing_classes = ['lumen']
test_metadata = MetadataCatalog.get(projectname+'test')
test_metadata.stuff_classes = ['lumen']
test_metadata.thing_classes = ['lumen']

cfg = get_cfg()
cfg.merge_from_file('/workspace/oct_ca_seg/runsaves/initPawsey/initialOCTPawsey_model_mask_rcnn_R_50_FPN_3x.yaml')
cfg.OUTPUT_DIR = ('/workspace/oct_ca_seg/runsaves/'+run_num+'_pawsey')

with open(cfg.OUTPUT_DIR +'/' + 'configD2.yaml', 'w') as file:
    file.write(cfg.dump())
    
iters = 10000
bs = 5
cfg.SOLVER.MAX_ITER = iters 
cfg.SOLVER.IMS_PER_BATCH = bs
cfg.SOLVER.CHECKPOINT_PERIOD = 50000

with mlflow.start_run():
    mlflow.log_param('iters',str(iters))
    mlflow.log_param('bs',str(bs))
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)


    coco_ev = COCOEvaluator(projectname+"valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
    OCT_ev = OCT_Evaluator(validCOCO)

    evaluators = DatasetEvaluators([coco_ev, OCT_ev])
    val_loader = build_detection_test_loader(cfg, projectname+"valid")
    results = inference_on_dataset(predictor.model, val_loader, evaluators)

    save_results(results, Path(cfg.OUTPUT_DIR)/'results.json')
    
    stats = {'u_dice': np.mean(list(results['dices'].values())),
             'u_spec': np.mean(list(results['specs'].values())),
             'u_sens': np.mean(list(results['sens'].values())),
             'u_acc': np.mean(list(results['accs'].values()))}
             
    mlflow.log_metrics(stats)
    mlflow.pytorch.log_model(trainer.model, registered_model_name='first mlflow try')
    mlflow.pytorch.save_model(trainer.model)

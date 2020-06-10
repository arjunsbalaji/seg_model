#!/usr/bin/env python3

from exp.nb_detectron import *
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader

'''
This python file takes 4 CLI arguments run_num (str), iters (int), bs (int), size (int). Defaults listed below.
bs 0-8 usually. iters 5k-10k, size 0 for medium set, 1 for full
'''
try:
    run_num = sys.argv[1] 
    iters = int(sys.argv[2])
    bs = int(sys.argv[3])
    size = str(sys.argv[4])
except:
    iters=1000
    bs=1
    run_num='NA'
    size = 0

#anno_file_name = 'medium_set_annotations.json' #full is annotations.josn med is 'medium_set_annotations.json'
if size == 0: anno_file_name = 'medium_set_annotations.json'
else: anno_file_name = 'annotations.json'

data_path = Path('/workspace/oct_ca_seg/COCOdata/')

projectname = 'OCT'

train_path = data_path/'train'
valid_path = data_path/'valid'

trainCOCO = COCO(train_path/('images/'+anno_file_name))
validCOCO = COCO(valid_path/('images/'+anno_file_name))

for d in [train_path, valid_path]:
    DatasetCatalog.register(projectname + d.name,
                            lambda d=d: load_coco_json(d/('images/'+anno_file_name), d/'images', dataset_name=d.name))  #get_dicts(d.name))#
    MetadataCatalog.get(projectname+ d.name).set(stuff_classes=["lumen"])
    
train_metadata = MetadataCatalog.get(projectname+'train')
train_metadata.stuff_classes = ['lumen']
train_metadata.thing_classes = ['lumen']
train_metadata.json_file = str(train_path/('images/'+anno_file_name))

valid_metadata = MetadataCatalog.get(projectname+'valid')
valid_metadata.stuff_classes = ['lumen']
valid_metadata.thing_classes = ['lumen']
valid_metadata.json_file = str(valid_path/('images/'+anno_file_name))

checkpoint = '/workspace/oct_ca_seg/runsaves/initPawsey/'

cfg = get_cfg()
cfg.merge_from_file(checkpoint + 'config.yaml')

    
cfg.SOLVER.MAX_ITER = iters 
cfg.SOLVER.IMS_PER_BATCH = bs
cfg.SOLVER.CHECKPOINT_PERIOD = 50000
cfg.DATASETS.TRAIN = (projectname+"train",)
cfg.DATASETS.TEST = (projectname+"valid",)

cfg.OUTPUT_DIR = ('/workspace/oct_ca_seg/runsaves/'+run_num+'_pawsey')

cfg.MODEL.WEIGHTS = os.path.join(checkpoint, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # set the testing threshold for this model

train_data_loader = build_detection_train_loader(cfg, mapper=mapper)
test_data_loader = build_detection_test_loader(cfg, projectname+"valid", mapper=mapper)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


with open(cfg.OUTPUT_DIR +'/' + 'configD2.yaml', 'w') as file:
    file.write(cfg.dump())


tracking_uri = 'file:/workspace/oct_ca_seg/runsaves/mlruns'
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run():
    
    mlflow.log_param('iters',iters)
    mlflow.log_param('bs',bs)
    
    client = MlflowClient()
    mlflow_run_id = mlflow.active_run().info.run_id
    
    
    #mlflow.set_tracking_uri(cfg.OUTPUT_DIR+'/mlruns')
    #mlflow.set_tracking_uri("http://localhost:5000")
    
    print(mlflow.get_tracking_uri())
    trainer = DefaultTrainer(cfg) 
    trainer.build_train_loader = train_data_loader
    trainer.build_test_loader = test_data_loader
    trainer.resume_or_load(resume=False)
    trainer.train()


    predictor = DefaultPredictor(cfg)


    coco_ev = COCOEvaluator(projectname+"valid", cfg, False, output_dir=cfg.OUTPUT_DIR)
    OCT_ev = OCT_Evaluator(validCOCO, 0.7)

    evaluators = DatasetEvaluators([coco_ev, OCT_ev])
    val_loader = build_detection_test_loader(cfg, projectname+"valid")
    results = inference_on_dataset(predictor.model, val_loader, evaluators)

    save_results(dict(results), cfg.OUTPUT_DIR+'/results.json')
    
    stats = {'u_dice': np.mean(list(results['dices'].values())),
             'u_spec': np.mean(list(results['specs'].values())),
             'u_sens': np.mean(list(results['sens'].values())),
             'u_acc': np.mean(list(results['accs'].values()))}
    
    
    mlflow.log_metrics(stats)
    mlflow.log_artifact(cfg.OUTPUT_DIR+'/results.json')
    
    mlflow.pytorch.log_model(trainer.model, 'model_mlflow_log')
    #mlflow.pytorch.save_model(trainer.model, 'model_mlflow_save')

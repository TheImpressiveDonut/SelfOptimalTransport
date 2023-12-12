python run.py exp.name=debug method=protonet dataset=tabula_muris
python run.py exp.name=cross_val_baseline_lambda method=protonet dataset=tabula_muris sot=true lambda_=1 &&
python run.py exp.name=cross_val_baseline_lambda method=protonet dataset=tabula_muris sot=true lambda_=10 &&
python run.py exp.name=cross_val_baseline_lambda method=protonet dataset=tabula_muris sot=true lambda_=20 &&

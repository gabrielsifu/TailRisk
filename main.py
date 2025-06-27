from Config.config import sett


if __name__ == '__main__':
    if sett.DataEngineer.execute:
        from Classes.data_engineer import DataEngineer
        # Extract data, Transform, Clean NaN, and Load
        # Create Features, Normalize, Divide Periods
        dc = DataEngineer(sett)
        dc.save_clean_data()

    if sett.DataScientist.execute:
        # Model Trainer
        from Classes.data_scientist import DataScientist
        ds = DataScientist(sett)
        ds.fit()

    if sett.MLOps.execute:
        # Machine Learning Operations
        from Classes.ml_ops import MLOps
        mlo = MLOps(sett)
        mlo.predict()

    if sett.Evaluator.execute:
        # Evaluator
        from Classes.evaluator import Evaluator
        e = Evaluator(sett)
        e.evaluate()


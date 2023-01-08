import experiments
import torch.distributed


def run_other_tests():
    # tests.test_basic_stats_4games()
    # tests.test_data_cleaner()
    # tests.test_csv_builder()
    # tests.test_point_differential()
    # tests.test_per_game_up_to_stats()
    # tests.test_small_dataset_NN()
    # tests.test_build_labeled_data()
    pass


if __name__ == '__main__':
    torch.manual_seed(42)
    # GameCSVBuilder.build_188_dim_data(True)
    # experiments.experiment_2()
    # experiments.experiment_3()
    # experiments.experiment_4()
    # experiments.experiment_5()
    # experiments.experiment_6()
    # experiments.experiment_7()
    # experiments.experiment_8()
    # experiments.experiment_9()
    # experiments.experiment_10()
    # experiments.experiment_11()
    experiments.experiment_12(False)
#    experiments.experiment_12(is_classification=True)
#    experiments.experiment_13()
    pass

from reader_yesno_continual_learning_effect_base import main_cl_base_model
from reader_yesno_continual_learning_effect_enhenced import main_cl_enhanced_context_model

if __name__ == '__main__':
    print("Starting Base Model Training Continual Learning")
    main_cl_base_model()
    print("Starting Enhanced Context model training")
    print(10*"_")
    print(10*"_")
    main_cl_enhanced_context_model()

from reader_yesno_base_models_ import main_base_model
from reader_yesno_enhanced_context import main_enhanced_context
from reader_yes_no_with_enhanced_if_size_not_big import main_enhanced_context_if_not_big
from reader_yes_no_with_enhanced_and_not_enhanced import main_enhanced_context_both_enhanced_and_not

if __name__ == '__main__':
    print("Starting Enhanced Context model training if needed")
    main_enhanced_context_if_not_big()

    print("Starting Enhanced Context model training and not enhanced")
    print(10 * "_")
    print(10 * "_")
    main_enhanced_context_both_enhanced_and_not()


    print("Starting Enhanced Context model training")
    print(10 * "_")
    print(10 * "_")
    main_enhanced_context()

    print(10 * "_")
    print(10 * "_")
    print("Starting Base Model Training")
    main_base_model()



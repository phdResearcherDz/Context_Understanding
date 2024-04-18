from reader_yesno_base_models_ import main_base_model
from reader_yesno_enhanced_context import main_enhanced_context

if __name__ == '__main__':
    print("Starting Base Model Training")
    main_base_model()
    print("Starting Enhanced Context model training")
    print(10*"_")
    print(10*"_")
    main_enhanced_context()

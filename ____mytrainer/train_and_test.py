from reader_yesno_base_models_ import main_base_model
from reader_yesno_enhanced_context import main_enhanced_context


from reader_yesno_cl_enhanced import main_cl_enhanced
from reader_yesno_cl_base import main_cl_base

if __name__ == '__main__':
    print("Without Continual Learning")
    main_base_model()
    main_enhanced_context()
    
    print("Continual Learning tests")
    main_cl_enhanced()
    main_cl_base()
    
    
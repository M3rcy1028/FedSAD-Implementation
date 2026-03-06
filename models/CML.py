from utils import *
from arguments import get_args
from model_aae import TransformerAAE, FLClient

WEIGHT_PATH = "./cml/cml_vae_transformer_weights1.h5"
MATRIX_PATH = "./cml/cml_cm1.png"
RESULT_PATH = "./cml/cml_server1.txt"
ROC_PATH = "./cml/cml_roc1.png"

EPOCHS = 300

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    # 1. Get dataset
    X_train_scaled, X_test_scaled, y_test = get_datasets_nsl()

    # 2. Define and compile the global(central) model
    input_dim = X_train_scaled.shape[1]
    global_model = TransformerAAE(input_dim)
    global_model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    _ = global_model(tf.zeros((1, input_dim)), prior_labels=tf.zeros((1,1)))

    # 3. Train the central model using the entire training dataset
    print(f"Starting Centralized ML training with {EPOCHS} epochs...")
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 훈련 데이터를 X_train_scaled로 수정
    history = global_model.fit(
        X_train_scaled, X_train_scaled,
        epochs=EPOCHS,
        batch_size=args.batch_size,
        shuffle=True,
        validation_split=0.3,
        callbacks=[early_stopping]
    )

    # 4. Evaluate the trained central model
    eval_server(
        global_model,
        X_train_scaled,
        X_test_scaled,
        y_test,
        result_path=RESULT_PATH,
        matrix_path=MATRIX_PATH,
        roc_path=ROC_PATH
    )

    # save_and_plot_history(
    #     history, 
    #     csv_path="./cml/cml_history", 
    #     png_path="./cml/cml_history.png"
    # )

    # 5. Save the trained model weights
    global_model.save_weights(WEIGHT_PATH)
    print(f"\nModel weights saved to {WEIGHT_PATH}")

if __name__ == "__main__":
    main()

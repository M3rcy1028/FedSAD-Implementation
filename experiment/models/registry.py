"""Model names and supported input representations, loaded only when used."""
MODELS = ("taae", "aelstm", "cnnlstm", "mvids", "mmfewshotsids")
TABULAR = ("kdd99", "nsl-kdd", "insdn", "cic2018", "unsw-nb15", "genis")
SUPPORT = {"taae": (*TABULAR, "gotham", "itu-ulak"),
           "aelstm": TABULAR, "cnnlstm": TABULAR,
           "mvids": ("cic2018", "unsw-nb15"),
           "mmfewshotsids": ("cic2018", "unsw-nb15")}


def build(config, shape):
    from experiment.backends import load
    import tensorflow as tf
    if config["model"] == "taae":
        module = load("taae_previous" if config["variant"] == "previous" else "taae_current",
                      config["batch_size"], config["percentile"], config["dropout"])
        model = module.TransformerAAE(input_dim=shape[-1], latent_dim=config["latent_dim"],
                                      dropout_rate=config["dropout"])
        model(tf.zeros((1, *shape)), prior_labels=tf.zeros((1, 1)))
        return model
    elif config["model"] == "aelstm":
        module = load("models.aelstm")
        model = module.AE_LSTM(input_dim=shape[-1], timesteps=shape[0], features=shape[1],
                              latent_dim=config["latent_dim"], dropout_rate=config["dropout"])
    else:
        module = load("models.cnnlstm")
        model = module.CNN_LSTM(timesteps=shape[0], features=shape[1])
    model(tf.zeros((1, *shape)))
    return model

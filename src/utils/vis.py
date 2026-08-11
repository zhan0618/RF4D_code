import matplotlib
import matplotlib.pyplot as plt


def save_loss_plot(loss, img_dir, args, global_step):
    matplotlib.use("Agg")
    name = args.name.strip("\"")
    img_path = img_dir / f"{name}_loss-global_step{global_step}.png"

    fig, ax = plt.subplots()
    for loss_term, values in loss.items():
        if values:
            ax.plot(values, label=loss_term)

    ax.set_xlabel("Global Step")
    ax.set_title(f"{name}: Loss")
    if any(values for values in loss.values()):
        ax.legend(loc="upper right")
    fig.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

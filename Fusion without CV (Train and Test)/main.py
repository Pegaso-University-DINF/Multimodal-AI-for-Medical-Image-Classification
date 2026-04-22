# from dataset import get_loaders
# from model import MultimodalClassifier
# from train import train_model
# import torch

# if __name__ == "__main__":
#     csv_file = "multimodal_dataset.csv" 
#     batch_size = 8
#     num_epochs = 10
#     lr = 1e-4

#     # Load data
#     train_loader, val_loader, num_classes = get_loaders(csv_file, batch_size=batch_size, num_workers=0)

#     # Initialize model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MultimodalClassifier(num_classes=num_classes, text_model_name='bert-base-uncased', image_model_name='resnet101')

#     # Train
#     trained_model = train_model(model, train_loader, val_loader, device, num_classes, num_epochs=num_epochs, lr=lr)

#     # Save
#     torch.save(trained_model.state_dict(), "multimodal_model_101.pth")
#     print("Multimodal model saved as multimodal_model.pth")

#++++++++++++++++++++++Modality Attention+++++++++++++++++++++++
# from dataset import get_loaders
# from model import MultimodalClassifier
# from train import train_model
# import torch

# if __name__ == "__main__":

#     csv_file = "multimodal_dataset.csv"
#     batch_size = 8
#     num_epochs = 10
#     lr = 1e-4

#     # Load data
#     train_loader, val_loader, num_classes = get_loaders(
#         csv_file,
#         batch_size=batch_size,
#         num_workers=0
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # IMPORTANT: ResNet50 only
#     model = MultimodalClassifier(
#         num_classes=num_classes,
#         text_model_name='bert-base-uncased'
#     )

#     trained_model = train_model(
#         model,
#         train_loader,
#         val_loader,
#         device,
#         num_classes,
#         num_epochs=num_epochs,
#         lr=lr
#     )

#     torch.save(trained_model.state_dict(),
#                "multimodal_model_resnet50_attention.pth")

#     print("Modality Attention model saved successfully.")


#++++++++++++++++Gated Fusion+++++++++++++++++++++++
from dataset import get_loaders
from model import MultimodalGatedClassifier
from train import train_model
import torch

if __name__ == "__main__":
    csv_file = "multimodal_dataset.csv" 
    batch_size = 8
    num_epochs = 10
    lr = 1e-4

    # Data loaders
    train_loader, val_loader, num_classes = get_loaders(csv_file, batch_size=batch_size, num_workers=0)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalGatedClassifier(num_classes=num_classes, text_model_name='bert-base-uncased')

    # Train
    trained_model = train_model(model, train_loader, val_loader, device, num_classes, num_epochs=num_epochs, lr=lr)

    # Save
    torch.save(trained_model.state_dict(), "multimodal_model_gated_resnet50.pth")
    print("Multimodal Gated Fusion model saved as multimodal_model_gated_resnet50.pth")


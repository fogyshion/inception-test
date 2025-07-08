from torchvision.models import inception_v3
from plato.servers import fedavg
from plato.clients import simple
from trainer import Trainer
from functools import partial
from CustomDataSource import CustomDataSource
from plato.datasources import tiny_imagenet

def main():
    # datasource = tiny_imagenet.DataSource()
    datasource = CustomDataSource
    model = partial(inception_v3, num_classes=200, aux_logits=True)
    trainer = Trainer
    client = simple.Client(model=model,datasource=datasource, trainer=trainer)
    server = fedavg.Server(model=model, datasource=datasource, trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()

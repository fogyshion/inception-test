from plato.servers import fedavg
from plato.clients import simple
from trainer import Trainer


def main():
    trainer = Trainer
    client = simple.Client(trainer=trainer)
    server = fedavg.Server(trainer=trainer)
    server.run(client)

if __name__ == "__main__":
    main()

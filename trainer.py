from plato.trainers import basic
import torch

class Trainer(basic.Trainer):
    def perform_forward_and_backward_passes(self, config, examples, labels):
        """
        重载父类方法，针对inception_v3修改
        """
        self.optimizer.zero_grad()
        outputs = self.model(examples)

        if config["model_name"] == "inception_v3":
            # 修改代码（针对inception_v3）
            logits, aux_logits = outputs.logits, outputs.aux_logits
            # print("logits的shape为：", logits.shape)
            # print("aux_logits的shape为：", aux_logits.shape)
            loss = self._loss_criterion(logits, labels) + 0.3 * self._loss_criterion(aux_logits, labels)
        else:
            # 原始代码
            loss = self._loss_criterion(outputs, labels)

        self._loss_tracker.update(loss, labels.size(0))

        if "create_graph" in config:
            loss.backward(create_graph=config["create_graph"])
        else:
            loss.backward()

        self.optimizer.step()

        return loss


    def test_model(self, config, testset, sampler=None, **kwargs):
        """
        重载父类方法，针对inception_v3修改
        """
        batch_size = config["batch_size"]

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler
        )

        correct = 0
        total = 0

        self.model.to(self.device)
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(self.device), labels.to(self.device)

                outputs = self.model(examples)

                outputs = self.process_outputs(outputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
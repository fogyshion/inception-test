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
            # 在原文中，总的损失为 1*主分类器损失 + 0.3*辅助分类器损失
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
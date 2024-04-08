"""
this version is modified from https://github.com/dkumazaw/mobilenetv3-pytorch/
"""

import time
import torch
import utils


class BaseTrainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, train_loader, valid_loader, test_loader,
                 epochs, logger, model_save_dir):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model_save_dir = model_save_dir
        self.logger = logger
        self.epochs = epochs

        self.model = self.model.to(device, non_blocking=True)


class ClassifierTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, scheduler,
                 device, train_loader, valid_loader, test_loader,
                 epochs, logger, model_save_dir):
        super(ClassifierTrainer, self).__init__(
            model, criterion, optimizer, scheduler,
            device, train_loader, valid_loader, test_loader,
            epochs, logger, model_save_dir
        )

    def train(self):
        """Trains the model for epochs"""
        best_top1_acc = 0

        for epoch in range(self.epochs):
            self.logger.info('epoch {}, lr {}'.format(
                epoch, self.scheduler.get_last_lr()
            ))
            start_time = time.time()

            # Training
            train_top1_acc, train_top3_acc, train_loss = self._train_epoch(
                epoch)
            self.logger.info(
                'train_top1_acc {:.5f}, train_top3_acc {:.5f}, train_loss {:.5f}'.format(
                    train_top1_acc, train_top3_acc, train_loss
                )
            )

            # Validation
            valid_top1_acc, valid_top3_acc, valid_loss = self._valid_epoch(
                epoch)
            self.logger.info(
                'valid_top1_acc {:.5f}, valid_top3_acc {:.5f}, valid_loss {:.5f}'.format(
                    valid_top1_acc, valid_top3_acc, valid_loss
                )
            )

            elapsed = time.time() - start_time
            self.logger.info('Took {} seconds'.format(elapsed))

            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True
            else:
                is_best = False

            # Create a checkpoint
            state = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            utils.save_checkpoint(
                state,
                is_best,
                self.model_save_dir,
                epoch
            )

    def _train_epoch(self, epoch: int):
        """Trains the model for one epoch"""
        total_loss = utils.AveTracker()
        top1_acc = utils.AveTracker()
        top3_acc = utils.AveTracker()
        self.model.train()

        for step, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            prec1, prec3 = utils.accuracy(logits, y, topk=(1, 3))
            n = x.size(0)
            total_loss.update(loss.item(), n)
            top1_acc.update(prec1.item(), n)
            top3_acc.update(prec3.item(), n)

            self.scheduler.step()

            if step != 0 and step % 100 == 0:
                self.logger.info('train %d %e %f %f', step,
                                 total_loss.average, top1_acc.average, top3_acc.average)

        return top1_acc.average, top3_acc.average, total_loss.average

    def _valid_epoch(self, epoch, phase='valid'):
        """Runs validation phase on either the validation or test set"""
        total_loss = utils.AveTracker()
        top1_acc = utils.AveTracker()
        top3_acc = utils.AveTracker()
        self.model.eval()

        if phase == 'test':
            self.logger.info('Running test set inference...')

        loader = self.valid_loader if phase == 'valid' else self.test_loader

        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                prec1, prec3 = utils.accuracy(logits, y, topk=(1, 3))
                n = x.size(0)
                total_loss.update(loss.item(), n)
                top1_acc.update(prec1.item(), n)
                top3_acc.update(prec3.item(), n)

                if step != 0 and step % 100 == 0:
                    self.logger.info(
                        '{} {:04d} {:e} {:f} {:f}'.format(
                            phase, step, total_loss.average, top1_acc.average, top3_acc.average
                        )
                    )

        return top1_acc.average, top3_acc.average, total_loss.average

    def validate(self):
        """Runs inference on test set to get the final performance metrics"""
        # Load the best performing model first
        best_state = utils.load_best_model_state_dict(self.model_save_dir)

        self.model.load_state_dict(
            best_state['model']
        )

        # Run validation on test set
        test_top1_acc, test_top3_acc, _ = self._valid_epoch(
            epoch=-1,
            phase='test',
        )
        self.logger.info(
            'test_top1_acc {:.5f}, test_top3_acc {:.5f}'.format(
                test_top1_acc,
                test_top3_acc
            )
        )
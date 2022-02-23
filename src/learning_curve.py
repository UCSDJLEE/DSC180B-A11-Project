import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

def learning_curve(training_rmse, validation_rmse, best_epoch):
	'''
	Function to plot the learning curve of our NN jet mass regressor model
	Function will output lineplot with RMSE values on vertical axis and # of epochs on horizontal axis

	Parameters:
	training_rmse: list of training RMSE loss values
	validation_rmse: list of validation RMSE loss values
	best_epoch: # of epoch at which the smallest loss is recorded
	'''
	epoch = [x+1 for x in range(len(validation_rmse))]
	_ = sns.set(
		context='notebook', rc={
		'axes.spines.right':False,
		'axes.spines.top':False
		}, style='white')

	fig = plt.figure(figsize=(10,10))
	ax = fig.gca()

	sns.lineplot(x=epoch, y=training_rmse, color='blue', ax=ax, label='Train RMSE loss')
	sns.lineplot(x=epoch, y=validation_rmse, color='orange', ax=ax, label='Validation RMSE loss');
	ax.plot(best_epoch+1, validation_rmse[best_epoch], marker='*', markerSize=12, color='red', label='Best model saved at');

	_ = ax.legend(frameon=True)
	_ = ax.set_xlabel('# of Epoch')
	_ = ax.set_ylabel('RMSE')
	_ = ax.set_title('Jet mass NN-regressor learning curve', fontdict={
	    'size':15,
	    'weight':'bold'
	})

	return ax
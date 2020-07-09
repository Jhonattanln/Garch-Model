import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

df = pd.read_excel(r'C:\Users\Jhona\OneDrive\Área de Trabalho\Atualizado.xlsx', index_col='Data')

returns = df['CPLE6'].pct_change()*100

#####Especificar o modelo garch
basic_gm = arch_model(returns[1:], p=1, q=1, mean='constant', vol='GARCH', dist='skewt')
gm_result = basic_gm.fit()

####Display model
print(gm_result.summary())

plt.style.use('ggplot')
gm_result.plot()
plt.show()

####Estimando volatilidade 
gm_forecast = gm_result.forecast(horizon = 5)
print(gm_forecast.variance[-1:])

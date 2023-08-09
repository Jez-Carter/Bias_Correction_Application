# Plotting
import matplotlib.pyplot as plt

def plotting_quantile_mapping(ax,scenario):
    ax.plot(scenario['t'],scenario['ocomplete_realisation'],label='True Underlying Field',alpha=0.6)
    ax.plot(scenario['t'],scenario['ccomplete_realisation'],label='Biased Underlying Field',alpha=0.6)
    ax.scatter(scenario['ot'],scenario['odata'],label='In-Situ Observations',alpha=1.0,s=10,marker='x')
    ax.scatter(scenario['ct'],scenario['cdata'],label='Climate Model Output',alpha=1.0,s=10,marker='x')

    ax.plot(scenario['ct'],scenario['c_corrected'].mean(axis=0),
            label='Bias Corrected Output Expectation',
            color='k',alpha=0.6,linestyle='dotted')

    ax.fill_between(scenario['ct'],
                    scenario['c_corrected'].mean(axis=0)+3*scenario['c_corrected'].std(axis=0),
                    scenario['c_corrected'].mean(axis=0)-3*scenario['c_corrected'].std(axis=0),
                    interpolate=True, color='k',alpha=0.2,
                    label='Bias Corrected Output Uncertainty 3$\sigma$')
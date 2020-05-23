from matplotlib import pyplot as plt,
y,ears=[1950,1960,1970,1980,1990,2000,2010],
gdp=[300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]

plt.plot(years,gdp,color='green',marker='o',linestyle='solid')
plt.title("Nominal GDP")
plt.ylabel("Billions of $")
plt.show()


#막대 그래프(이산적)
movies=["Annie Hall","Ben_Hur","Casablnaca","Gandhi","West Side Story"]
num_oscars=[5,11,3,8,10]

plt.bar(range(len(movies)),num_oscars)
plt.title("My Favorite Movies")
plt.ylabel("#of Academy Awards")
plt.xticks(range(len(movies)),movies)
plt.show()


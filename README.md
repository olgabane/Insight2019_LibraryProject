# Insight2019_Library-Project
**Welcome to BiblioPal!** 
This is a porject created by Olga Minkina over 3 weeks as an Insight Data Science Fellow (Jan-Mar 2019). 

**What is BiblioPal?**

BiblioPal is a tool to determine how libraries’ resources, staffing and spending choices affect library usage.

**How can your library use BiblioPal?**
Use case #1: Your library has increased funding next year (congratulations!). How can you best allocate those funds to efficiently increase library usage in your community?

Use case #2: Your library funding is limited next year. Where can you remove resources such that it has minimal impact on library usage?

**The data:**

The data used in this project is publicly available from the Institute of Museum and Library Services, which surveys over 9,000 public libraries in the United States annually. I am using data for the last 7 years available (2010-2016). 

Does library usage vary among public libraries in the United States?

Yes, library usage varies widely (from <1 to 50) among public libraries in the United States.
Library usage is defined as Annual Visits/Local population throughout this project.

**How can your library increase library usage?**

Your libraries makes a lot of operational choices that may impact library usage:

1. Resources: collections offered, programs offered, other services provided (i.e. bookmobile, hours open). 
2. Staff: Number of staff, education level of staff.
3. Spending: Collection expenditure, staff expenditure.
But which choices actually impact library usage and to what degree?

**The model:**

I used a random forest regressor model to determine how resources, staffing and spending choices impact library usage. With this model, I can predict library usage within 1 point in 40% of cases and within 3 points in 79% of cases. Expenditure on print materials has the highest feature importance in this model. 

**Web application**

BiblioPal is available on the web at www.ominkina.com!

A library can use BiblioPal in the following way:
1. Choose the state where your library is located.
2. Choose your library.
3. Resource, Staff and Spending values are pre-populated with 2016 values (the latest year for which the data is available). Alter these values using up-down arrows or by directly typing new value within text box. 
4. Click Submit to see predicted usage and predicted visitor count.

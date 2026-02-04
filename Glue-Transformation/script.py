#!/usr/bin/env python
# coding: utf-8

# =========================================================
# 1. Glue setup & job parameters
# =========================================================

import sys
import re
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

from pyspark.sql.functions import (
    col, when, sum, round, regexp_replace,
    to_date, year, month, trim, upper
)
from pyspark.sql.types import IntegerType, DecimalType

args = getResolvedOptions(sys.argv, ["RAW_S3_PATH", "CLEAN_S3_PATH"])

RAW_S3_PATH = args["RAW_S3_PATH"]
CLEAN_S3_PATH = args["CLEAN_S3_PATH"]

sc = SparkContext.getOrCreate()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init("liquor-sales-cleaning-job-copy", args)

# =========================================================
# 2. Read RAW CSV
# =========================================================

df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(RAW_S3_PATH)

# =========================================================
# 3. Normalize column names
# =========================================================

df = df.toDF(*[re.sub(r"\s+", "_", c.strip()) for c in df.columns])

# =========================================================
# 4. Drop critical NULL rows
# =========================================================

df_clean = df.dropna(subset=[
    "Address", "City", "Zip_Code",
    "Volume_Sold_(Gallons)", "Volume_Sold_(Liters)",
    "Sale_(Dollars)", "Bottles_Sold",
    "State_Bottle_Retail", "Bottle_Volume_(ml)",
    "Pack", "Item_Number", "Vendor_Name",
    "Vendor_Number", "Category_Name", "Category"
])

df_clean = df_clean.drop("Store_Location")

# =========================================================
# 5. RENAME Invoice column ( FIX ADDED HERE )
# =========================================================

df_clean = df_clean.withColumnRenamed(
    "Invoice/Item_Number",
    "Invoice_Number"
)

# =========================================================
# 6. COUNTY & COUNTY NUMBER HANDLING
# =========================================================

county_data = [
    ("ADAIR",1),("ADAMS",2),("ALLAMAKEE",3),("APPANOOSE",4),("AUDUBON",5),
    ("BENTON",6),("BLACK_HAWK",7),("BOONE",8),("BREMER",9),("BUCHANAN",10),
    ("BUENA_VISTA",11),("BUTLER",12),("CALHOUN",13),("CARROLL",14),("CASS",15),
    ("CEDAR",16),("CERRO_GORDO",17),("CHEROKEE",18),("CHICKASAW",19),
    ("CLARKE",20),("CLAY",21),("CLAYTON",22),("CLINTON",23),("CRAWFORD",24),
    ("DALLAS",25),("DAVIS",26),("DECATUR",27),("DELAWARE",28),
    ("DES_MOINES",29),("DICKINSON",30),("DUBUQUE",31),("EMMET",32),
    ("FAYETTE",33),("FLOYD",34),("FRANKLIN",35),("FREMONT",36),
    ("GREENE",37),("GRUNDY",38),("GUTHRIE",39),("HAMILTON",40),
    ("HANCOCK",41),("HARDIN",42),("HARRISON",43),("HENRY",44),
    ("HOWARD",45),("HUMBOLDT",46),("IDA",47),("IOWA",48),("JACKSON",49),
    ("JASPER",50),("JEFFERSON",51),("JOHNSON",52),("JONES",53),
    ("KEOKUK",54),("KOSSUTH",55),("LEE",56),("LINN",57),("LOUISA",58),
    ("LUCAS",59),("LYON",60),("MADISON",61),("MAHASKA",62),
    ("MARION",63),("MARSHALL",64),("MILLS",65),("MITCHELL",66),
    ("MONONA",67),("MONROE",68),("MONTGOMERY",69),("MUSCATINE",70),
    ("O'BRIEN",71),("OSCEOLA",72),("PAGE",73),("PALO_ALTO",74),
    ("PLYMOUTH",75),("POCAHONTAS",76),("POLK",77),
    ("POTTAWATTAMIE",78),("POWESHIEK",79),("RINGGOLD",80),
    ("SAC",81),("SCOTT",82),("SHELBY",83),("SIOUX",84),
    ("STORY",85),("TAMA",86),("TAYLOR",87),("UNION",88),
    ("VAN_BUREN",89),("WAPELLO",90),("WARREN",91),("WASHINGTON",92),
    ("WAYNE",93),("WEBSTER",94),("WINNEBAGO",95),
    ("WINNESHIEK",96),("WOODBURY",97),("WORTH",98),("WRIGHT",99)
]

county_df = spark.createDataFrame(county_data, ["County", "County_Number_LKP"])

df_clean = df_clean.withColumn("County", upper(trim(col("County"))))

df_clean = df_clean.join(county_df, on="County", how="left")

df_clean = df_clean.withColumn(
    "County_Number",
    when(col("County_Number").isNull(), col("County_Number_LKP"))
    .otherwise(col("County_Number"))
).drop("County_Number_LKP")

# Final safety net
df_clean = df_clean.withColumn(
    "County", when(col("County").isNull(), "UNKNOWN").otherwise(col("County"))
)

df_clean = df_clean.withColumn(
    "County_Number", when(col("County_Number").isNull(), -1).otherwise(col("County_Number"))
)

# =========================================================
# 7. TYPE CASTING
# =========================================================

df_clean = df_clean \
    .withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) \
    .withColumn("Store_Number", col("Store_Number").cast(IntegerType())) \
    .withColumn("County_Number", col("County_Number").cast(IntegerType())) \
    .withColumn("Category", col("Category").cast(IntegerType())) \
    .withColumn("Vendor_Number", col("Vendor_Number").cast(IntegerType())) \
    .withColumn("Pack", col("Pack").cast(IntegerType())) \
    .withColumn("Bottle_Volume_(ml)", regexp_replace(col("Bottle_Volume_(ml)"), ",", "").cast(IntegerType())) \
    .withColumn("Bottles_Sold", regexp_replace(col("Bottles_Sold"), ",", "").cast(IntegerType())) \
    .withColumn("State_Bottle_Cost", regexp_replace(col("State_Bottle_Cost"), "[$,]", "").cast(DecimalType(10,2))) \
    .withColumn("State_Bottle_Retail", regexp_replace(col("State_Bottle_Retail"), "[$,]", "").cast(DecimalType(10,2))) \
    .withColumn("Sale_(Dollars)", regexp_replace(col("Sale_(Dollars)"), "[$,]", "").cast(DecimalType(12,2))) \
    .withColumn("Volume_Sold_(Liters)", regexp_replace(col("Volume_Sold_(Liters)"), ",", "").cast(DecimalType(10,2))) \
    .withColumn("Volume_Sold_(Gallons)", regexp_replace(col("Volume_Sold_(Gallons)"), ",", "").cast(DecimalType(10,2)))


df_clean = df_clean.filter(col("Date").isNotNull())

# =========================================================
# 8. FIX VOLUME NULLS (BUSINESS LOGIC)
# =========================================================

df_clean = df_clean.withColumn(
    "Volume_Sold_(Liters)",
    when(
        col("Volume_Sold_(Liters)").isNull() &
        col("Bottle_Volume_(ml)").isNotNull() &
        col("Bottles_Sold").isNotNull(),
        round(
            (col("Bottle_Volume_(ml)") * col("Bottles_Sold")) / 1000,
            2
        ).cast(DecimalType(10, 2))
    ).otherwise(col("Volume_Sold_(Liters)"))
)

df_clean = df_clean.withColumn(
    "Volume_Sold_(Gallons)",
    when(
        col("Volume_Sold_(Gallons)").isNull() &
        col("Volume_Sold_(Liters)").isNotNull(),
        round(
            col("Volume_Sold_(Liters)") * 0.264172,
            2
        ).cast(DecimalType(10, 2))
    ).otherwise(col("Volume_Sold_(Gallons)"))
)


# =========================================================
# 9. METRICS
# =========================================================

df_clean = df_clean \
    .withColumn("profit_per_bottle", round(col("State_Bottle_Retail") - col("State_Bottle_Cost"), 2)) \
    .withColumn(
        "Revenue_per_Bottle",
        when(col("Bottles_Sold") > 0,
             round(col("Sale_(Dollars)") / col("Bottles_Sold"), 2))
        .otherwise(0)
    ) \
    .withColumn(
        "total_profit",
        round(col("profit_per_bottle") * col("Bottles_Sold"), 2)
    )

# =========================================================
# 10. PARTITION & WRITE
# =========================================================

(
    df_clean
    .withColumn("year", year(col("Date")).cast("string"))
    .withColumn("month", month(col("Date")).cast("string"))
    .repartition("year", "month")
    .write
    .mode("overwrite")
    .format("parquet")
    .partitionBy("year", "month")
    .save(CLEAN_S3_PATH)
)


print("RAW:", RAW_S3_PATH)
print("CLEAN:", CLEAN_S3_PATH)


# =========================================================
# 11. FINISH JOB
# =========================================================

job.commit()

from os.path import join

from pandas import DataFrame, read_csv

from tabstar_paper.datasets.curation_objects import CuratedTarget, CuratedFeature
from tabstar_paper.datasets.objects import SupervisedTask, FeatureType

'''
Dataset Name: selener/consumer-complaint-database/
====
Examples: 1282348
====
URL: https://www.kaggle.com/selener/consumer-complaint-database
====
Description: 
Consumer Complaints Dataset This dataset contains records of consumer complaints filed against financial service
companies, featuring structured fields such as product type, state, and submission method, alongside rich textual
attributes like sub-product, issue, sub-issue, and company name. The prediction target is the company’s response to the
complaint, consolidated into four meaningful classes: CLOSED WITH EXPLANATION, CLOSED WITH NON-MONETARY
RELIEF, CLOSED WITH MONETARY RELIEF, AND CLOSED WITHOUT RELIEF. This focused setup maintains a
challenging multi-class classification task while allowing for a fair assessment of how textual embeddings can enhance
tabular model performance in real-world customer service scenarios. The original dataset has approx. 1.42m rows,
which we randomly downsample to 100k to include a fair representation of rare classes while keeping the size feasible
for training.


About Dataset
Context
These are real world complaints received about financial products and services. Each complaint has been labeled with a specific product; therefore, this is a supervised text classification problem. With the aim to classify future complaints based on its content, we used different machine learning algorithms can make more accurate predictions (i.e., classify the complaint in one of the product categories)

Content
The dataset contains different information of complaints that customers have made about a multiple products and services in the financial sector, such us Credit Reports, Student Loans, Money Transfer, etc.
The date of each complaint ranges from November 2011 to May 2019.

Acknowledgements
This work is considered a U.S. Government Work. The dataset is public dataset and it was downloaded from
https://catalog.data.gov/dataset/consumer-complaint-database
on 2019, May 13.

Inspiration
This is a sort of tutorial for beginner

====
Target Variable: Company response to consumer (object, 8 distinct): ['Closed with explanation', 'Closed with non-monetary relief', 'Closed with monetary relief', 'Closed without relief', 'Closed', 'In progress', 'Untimely response', 'Closed with relief']
====
Features:

Date received (datetime64[ns], 0 distinct): ['2017-09-08 00:00:00', '2017-09-09 00:00:00', '2017-01-19 00:00:00', '2017-01-20 00:00:00', '2017-09-13 00:00:00', '2018-04-05 00:00:00', '2017-09-12 00:00:00', '2018-04-10 00:00:00', '2017-09-11 00:00:00', '2017-09-14 00:00:00']
Product (object, 18 distinct): ['Mortgage', 'Debt collection', 'Credit reporting, credit repair services, or other personal consumer reports', 'Credit reporting', 'Credit card', 'Bank account or service', 'Student loan', 'Credit card or prepaid card', 'Checking or savings account', 'Consumer Loan']
Sub-product (object, 76 distinct): ['Credit reporting', 'Checking account', 'Other mortgage', 'Conventional fixed mortgage', 'I do not know', 'Other (i.e. phone, health club, etc.)', 'General-purpose credit card or charge card', 'FHA mortgage', 'Other debt', 'Conventional home mortgage']
Issue (object, 167 distinct): ['Incorrect information on your report', 'Loan modification,collection,foreclosure', 'Incorrect information on credit report', 'Loan servicing, payments, escrow account', "Cont'd attempts collect debt not owed", "Problem with a credit reporting company's investigation into an existing problem", 'Attempts to collect debt not owed', 'Account opening, closing, or management', 'Communication tactics', 'Improper use of your report']
Sub-issue (object, 218 distinct): ['Information belongs to someone else', 'Account status', 'Their investigation did not fix an error on your report', 'Debt is not mine', 'Information is not mine', 'Account status incorrect', 'Debt was paid', 'Account information incorrect', 'Debt is not yours', 'Not given enough info to verify debt']
Consumer complaint narrative (object, 366941 distinct): ['There are many mistakes appear in my report without my understanding.', 'Equifax mishandled my information which has led to a breach that puts myself and millions of others at potential risk. I am extremely disappointed with how equifax has handled reporting this breach. Very little was done to notify the public for nearly a month after the breach was detected. I received no email, letter, or phone call and instead had to discover it via social media.', 'I have been a victim of identity theft.', 'I am filing this complaint because Experian has ignored my request to provide me with the documents that their company has on file that was used to verify the accounts I disputed. Being that they have gone past the 30 day mark and can not verify these accounts, under Section 611 ( 5 ) ( A ) of the FCRA - they are required to " ... promptly delete all information which can not be verified \'\' that I have disputed. Please resolve this manner as soon as possible. Thank you.', '( a ) Block. Except as otherwise provided in this section, a consumer reporting agency shall block the reporting of any information in the file of a consumer that the consumer identifies as information that resulted from an alleged identity theft, not later than 4 business days after the date of receipt by such agency of ( 1 ) appropriate proof of the identity of the consumer ; ( 2 ) a copy of an identity theft report ; ( 3 ) the identification of such information by the consumer ; and ( 4 ) a statement by the consumer that the information is not information relating to any transaction by the consumer.\n\n( b ) Notification. A consumer reporting agency shall promptly notify the furnisher of information identified by the consumer under subsection ( a ) of this section ( 1 ) that the information may be a result of identity theft ; ( 2 ) that an identity theft report has been filed ; ( 3 ) that a block has been requested under this section ; and ( 4 ) of the effective dates of the block.\n\n( c ) Authority to decline or rescind.\n\n( 1 ) In general. A consumer reporting agency may decline to block, or may rescind any block, of information relating to a consumer under this section, if the consumer reporting agency reasonably determines that ( A ) the information was blocked in error or a block was requested by the consumer in error ; ( B ) the information was blocked, or a block was requested by the consumer, on the basis of a material misrepresentation of fact by the consumer relevant to the request to block ; or ( C ) the consumer obtained possession of goods, services, or money as a result of the blocked transaction or transactions.\n\n( 2 ) Notification to consumer. If a block of information is declined or rescinded under this subsection, the affected consumer shall be notified promptly, in the same manner as consumers are notified of the reinsertion of information under section 1681i ( a ) ( 5 ) ( B ) of this title.\n\n( 3 ) Significance of block. For purposes of this subsection, if a consumer reporting agency rescinds a block, the presence of information in the file of a consumer prior to the blocking of such information is not evidence of whether the consumer knew or should have known that the consumer obtained possession of any goods, services, or money as a result of the block.\n\n( d ) Exception for resellers.\n\n( 1 ) No reseller file. This section shall not apply to a consumer reporting agency, if the consumer reporting agency ( A ) is a reseller ; ( B ) is not, at the time of the request of the consumer under subsection ( a ) of this section, otherwise furnishing or reselling a consumer report concerning the information identified by the consumer ; and ( C ) informs the consumer, by any means, that the consumer may report the identity theft to the Bureau to obtain consumer information regarding identity theft.\n\n( 2 ) Reseller with file. The sole obligation of the consumer reporting agency under this section, with regard to any request of a consumer under this section, shall be to block the consumer report maintained by the consumer reporting agency from any subsequent use, if ( A ) the consumer, in accordance with the provisions of subsection ( a ) of this section, identifies, to a consumer reporting agency, information in the file of the consumer that resulted from identity theft ; and ( B ) the consumer reporting agency is a reseller of the identified information.\n\n( 3 ) Notice. In carrying out its obligation under paragraph ( 2 ), the reseller shall promptly provide a notice to the consumer of the decision to block the file. Such notice shall contain the name, address, and telephone number of each consumer reporting agency from which the consumer information was obtained for resale.\n\n( e ) Exception for verification companies. The provisions of this section do not apply to a check services company, acting as such, which issues authorizations for the purpose of approving or processing negotiable instruments, electronic fund transfers, or similar methods of payments, except that, beginning 4 business days after receipt of information described in paragraphs ( 1 ) through ( 3 ) of subsection ( a ) of this section, a check services company shall not report to a national consumer reporting agency described in section 1681a ( p ) of this title, any information identified in the subject identity theft report as resulting from identity theft.\n\n( f ) Access to blocked information by law enforcement agencies. No provision of this section shall be construed as requiring a consumer reporting agency to prevent a Federal, State, or local law enforcement agency from accessing blocked information in a consumer file to which the agency could otherwise obtain access under this subchapter.', 'Creditor is reporting accounts that are invalid and unverified. XXXX have investigated this account and they concluded the account was not matching my records of being assigned to me. I would like proof these accounts were disputed per FCRA/ FDCPA. This is causing me anguish and its slandering my good name. Please block this account from my report so I can breathe again.', 'I am filing this complaint because TransUnion has ignored my request to provide me with the documents that their company has on file that was used to verify the accounts I disputed. Being that they have gone past the 30 day mark and can not verify these accounts, under Section 611 ( 5 ) ( A ) of the FCRA - they are required to " ... promptly delete all information which can not be verified \'\' that I have disputed. Please resolve this manner as soon as possible. Thank you.', 'This company continues to report on my credit report after I sent them a letter telling them that this account was not mine and I have no idea what it is or who it belongs to! \n\nI asked for proof of a signed contract, I asked for a license to collect in my state, I asked for copies of all information referenced for this debt and still to date, I have not received anything but harassment from this company! \n\nTHIS IS NOT MY DEBT! \n\nI WANT THIS ACCOUNT REMOVED FROM MY CREDIT REPORT AND THIS COMPANY TO STOP CONTACTING ME IMMEDIATELY!', "After checking my credit I realize that I have been victim of identity theft, An identity theft report had been filed with the FTC and a police report will be filed. As require by FCRA any information reported as fraudulent should be removed from the consumer credit file. I'm reporting this as an effort to clear my name.", 'I am filing this complaint because Equifax has ignored my request to provide me with the documents that their company has on file that was used to verify the accounts I disputed. Being that they have gone past the 30 day mark and can not verify these accounts, under Section 611 ( 5 ) ( A ) of the FCRA - they are required to " ... promptly delete all information which can not be verified \'\' that I have disputed. Please resolve this manner as soon as possible. Thank you.']
Company public response (object, 10 distinct): ['Company has responded to the consumer and the CFPB and chooses not to provide a public response', 'Company believes it acted appropriately as authorized by contract or law', 'Company chooses not to provide a public response', 'Company believes the complaint is the result of a misunderstanding', 'Company disputes the facts presented in the complaint', 'Company believes complaint caused principally by actions of third party outside the control or direction of the company', 'Company believes complaint is the result of an isolated error', 'Company believes complaint represents an opportunity for improvement to better serve consumers', "Company can't verify or dispute the facts in the complaint", 'Company believes complaint relates to a discontinued policy or procedure']
Company (object, 5275 distinct): ['EQUIFAX, INC.', 'Experian Information Solutions Inc.', 'TRANSUNION INTERMEDIATE HOLDINGS, INC.', 'BANK OF AMERICA, NATIONAL ASSOCIATION', 'WELLS FARGO & COMPANY', 'JPMORGAN CHASE & CO.', 'CITIBANK, N.A.', 'CAPITAL ONE FINANCIAL CORPORATION', 'Navient Solutions, LLC.', 'OCWEN LOAN SERVICING LLC']
State (object, 63 distinct): ['CA', 'FL', 'TX', 'NY', 'GA', 'IL', 'NJ', 'PA', 'NC', 'OH']
ZIP code (object, 22591 distinct): ['300XX', '770XX', '330XX', '331XX', '606XX', '750XX', '334XX', '303XX', '945XX', '900XX']
Tags (object, 3 distinct): ['Servicemember', 'Older American', 'Older American, Servicemember']
Consumer consent provided? (object, 4 distinct): ['Consent provided', 'Consent not provided', 'Other', 'Consent withdrawn']
Submitted via (object, 6 distinct): ['Web', 'Referral', 'Phone', 'Postal mail', 'Fax', 'Email']
Date sent to company (datetime64[ns], 0 distinct): ['2017-09-08 00:00:00', '2017-09-09 00:00:00', '2017-01-19 00:00:00', '2017-09-13 00:00:00', '2017-01-20 00:00:00', '2017-09-14 00:00:00', '2017-01-24 00:00:00', '2018-04-10 00:00:00', '2019-04-02 00:00:00', '2019-04-16 00:00:00']
Timely response? (object, 2 distinct): ['Yes', 'No']
Consumer disputed? (object, 2 distinct): ['No', 'Yes']
'''

def load_df(dir_path: str) -> DataFrame:
    df_path = join(dir_path, "rows.csv")
    df = read_csv(df_path)
    return df



CONTEXT = "Real world complaints of consumer over financial issues"
TARGET = CuratedTarget(raw_name="Company response to consumer", task_type=SupervisedTask.MULTICLASS)
FEATURES = [CuratedFeature(raw_name="Date received", feat_type=FeatureType.DATE),
            CuratedFeature(raw_name="Date sent to company", feat_type=FeatureType.DATE),]
COLS_TO_DROP = ['Complaint ID']
IMAGE_FOLDER = None
LOADING_FUNC = load_df
PROCESSING_FUNC = None

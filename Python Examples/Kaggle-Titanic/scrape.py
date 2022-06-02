"""
Scrape Kaggle Competition Leaderboards
"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager  # Do this if you get Selenium webdriver errors

driver = webdriver.Chrome(ChromeDriverManager().install())  # I'm not sure if `ChromeDriverManager` has to be called once to install to computer, or each time a driver is instatiated.
driver.get("https://www.kaggle.com/competitions/titanic/leaderboard#")

xpathTable = r'class="km-list km-list--one-line"'
elTable = driver.find_elements_by_xpath(xpathTable)

# Table header HTML:
"""
<li class="sc-gSYDnn jcqGXt">
    <div class="sc-laZMeE jaMIyn sc-fArJxI iZaeuW"><span class="sc-kLojOw sc-iklJeh sc-xoZJL dfYLnb hbusih eshVbH"
            style="font-weight: 700;">#</span><span class="sc-kLojOw sc-iklJeh sc-fpPwup dfYLnb hbusih fnGzZo"
            style="font-weight: 700;">Team</span><span class="sc-kLojOw sc-iklJeh sc-jwtsuN dfYLnb hbusih klQLKp"
            style="font-weight: 700;">Members</span><span class="sc-kLojOw sc-iklJeh sc-eemifY dfYLnb hbusih gaEEDr"
            style="font-weight: 700;">Score</span><span class="sc-kLojOw sc-iklJeh sc-cgTEYn dfYLnb hbusih cdfZFq"
            style="font-weight: 700;">Entries</span><span class="sc-kLojOw sc-iklJeh sc-ieieiu dfYLnb hbusih hkWzeY"
            style="font-weight: 700;">Last</span><span class="sc-kLojOw sc-iklJeh sc-bMQVVx dfYLnb hbusih kUIYsa"
            style="font-weight: 700;">Code</span></div>
    <div class="sc-hTRkXV ggURXR"></div>
</li>
"""
# Note the following of the `class` attribute values:
# "sc-kLojOw sc-iklJeh sc-" is common to all columns
# "xoZJL" is common to first column
# "fpPwup" is common to the second column
# "jwtsuN" is common to the third column
# "eemifY" is common to the 4th column
# "cgTEYn" is common to the 5th column
# "ieieiu" is common to the 6th column
# "bMQVVx" is common to the 7th column
# "dfYLnb hbusih" is common to all columns
# You can use the above facts to scrape data.
# Table row HTML:
"""
<li class="sc-gSYDnn jcqGXt">
    <div class="sc-laZMeE jaMIyn sc-kJrGqu fVAHCq"><span
            class="sc-kLojOw sc-iklJeh sc-xoZJL eGLYxv hbusih eshVbH">1</span><span
            class="sc-kLojOw sc-iklJeh sc-fpPwup eGLYxv hbusih fnGzZo"><span
                class="sc-kLojOw sc-iklJeh sc-kSyTMx eGLYxv hbusih kJaaHR">no name</span></span><span
            class="sc-kLojOw sc-iklJeh sc-jwtsuN eGLYxv hbusih klQLKp">
            <div class="sc-eGEtOM jBsTgx" style="flex-wrap: wrap; gap: 8px;"><a href="/andrej0marinchenko"
                    class="sc-FRrlG cIjgB" aria-label="andrej0marinchenko">
                    <div size="24" title="andrej0marinchenko" class="sc-fXazdy loYZUm"
                        style="background-image: url(&quot;https://storage.googleapis.com/kaggle-avatars/thumbnails/7662674-kg.jpg&quot;);">
                    </div><svg width="32" height="32" viewBox="0 0 32 32">
                        <circle r="15" cx="16" cy="16" fill="none" stroke-width="2" style="stroke: rgb(241, 243, 244);">
                        </circle>
                        <path d="M 7.183221215612905 28.135254915624213 A 15 15 0 1 0 16 1" fill="none" stroke-width="2"
                            style="stroke: rgb(101, 31, 255);"></path>
                    </svg>
                </a></div>
        </span><span class="sc-kLojOw sc-iklJeh sc-eemifY eGLYxv hbusih gaEEDr">
            <div class="sc-hKfvfE EnRhI">
                <div style="height: 18px; width: 18px; margin-right: 8px;"></div>1.00000
            </div>
        </span><span class="sc-kLojOw sc-iklJeh sc-cgTEYn eGLYxv hbusih cdfZFq">214</span><span
            class="sc-kLojOw sc-iklJeh sc-ieieiu eGLYxv hbusih hkWzeY">
            <div class="sc-TtZnY kkpQHT" data-tip="true"
                data-for="tooltip_components_a51971b8-971f-4567-8ef6-d9c463826317" currentitem="false">3h<div
                    class="__react_component_tooltip t46429cc3-2d88-4163-8e9c-96217b10107c place-top type-dark"
                    id="tooltip_components_a51971b8-971f-4567-8ef6-d9c463826317" data-id="tooltip">
                    <style aria-hidden="true">
                        "omitted by herman"
                    </style>
                    <div class="sc-jHNicF kXWezy">Last Submission: 5/18/2022, 12:19:39 PM EDT</div>
                </div>
            </div>
        </span><span class="sc-kLojOw sc-iklJeh sc-bMQVVx eGLYxv hbusih kUIYsa"></span></div>
    <div class="sc-hTRkXV ggURXR"></div>
</li>
"""
# At the top of the page Kaggle tells you how many teams there are, which is equal to the number of rows in the table. The number of teams is in the note with the following attribute and value:
# class="horizontal-list-item horizontal-list-item--bullet horizontal-list-item--default"
# So just keep scrolling down until you hit that number of rows.
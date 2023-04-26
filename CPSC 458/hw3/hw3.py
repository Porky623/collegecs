# %% [markdown]
# #### Feature options
#  - Monitor
#    - Refresh rate
#      - 60 Hz
#      - 120+ Hz
#    - Brightness/color quality
#      - Medium quality
#      - High quality
#  - CPU
#    - Single-core performance
#      - Low performance
#      - Medium-high performance
#    - Number of cores
#      - 1-4 cores
#      - 6+ cores
#  - GPU
#    - GPU RAM
#      - Low memory (<6 GB)
#      - Medium memory (6-8 GB)
#      - High memory (>8 GB)
#  - Price
#    - Budget (<$1000)
#    - Moderate ($1000-$1800)
#    - Luxury (>$1800)
#  - Storage (i.e. SSD)
#    - Low storage (<512 GB)
#    - Medium storage (512 GB-1 TB)
#    - High storage (SSD + HDD, >1 TB)
#  - Memory (i.e. RAM)
#    - Low memory (<8 GB)
#    - Medium memory (8-12 GB)
#    - High memory (>12 GB)

# %%
class feature:
    count = 0
    features = {}

    def __init__(self, name):
        self.name: str = name.upper()
        if self.name not in feature.features:
            self.count = feature.count
            feature.count += 1
            feature.features[self.name] = self

    def __repr__(self):
        return f'feature({self.name})'

    def __str__(self):
        return f'<feature ({self.count}): {self.name}>'

    def __eq__(self, other):
        return self.name == other.name


fs: list[list[str]] = [["60 HZ REFRESH", "120+ HZ REFRESH"],
      ["MEDIUM BRIGHTNESS/COLOR QUALITY", "HIGH BRIGHTNESS/COLOR QUALITY"],
      ["LOW SINGLE-CORE PERFORMANCE", "MEDIUM-HIGH SINGLE CORE PERFORMANCE"],
      ["1-4 CORES", "6+ CORES"],
      ["LOW GPU MEMORY", "MEDIUM GPU MEMORY", "HIGH GPU MEMORY"],
      ["BUDGET", "MODERATE PRICE", "LUXURY"],
      ["LOW STORAGE", "MEDIUM STORAGE", "HIGH STORAGE"],
      ["LOW MEMORY", "MEDIUM MEMORY", "HIGH MEMORY"]]
for f in fs:
    for x in f:
        feature(x)
print(feature.features)


# %% [markdown]
# #### Device Options
# 8 are from Costco's website, the remaining 20 are arbitrarily made for diversity

# %%
class device:
    count = 0
    devices = {}

    def __init__(self, name, features):
        self.name: str = name.upper()
        if self.name not in device.devices:
            self.count = device.count
            device.count += 1
            device.devices[self.name] = self
            self.features = [feature.features[f] for f in features]

    def __repr__(self):
        return f'device({self.name})'

    def __str__(self):
        return f'<device ({self.count}): {self.name}>'

    def __eq__(self, other):
        return self.name == other.name


device("HP DESKTOP", [fs[0][0],
                      fs[1][0],
                      fs[2][0],
                      fs[3][0],
                      fs[4][0],
                      fs[5][0],
                      fs[6][2],
                      fs[7][1]])
device("MAC MINI", [fs[0][0],
                    fs[1][0],
                    fs[2][1],
                    fs[3][1],
                    fs[4][1],
                    fs[5][0],
                    fs[6][0],
                    fs[7][1]])
device("IMAC", [fs[0][0],
                fs[1][1],
                fs[2][1],
                fs[3][1],
                fs[4][0],
                fs[5][1],
                fs[6][0],
                fs[7][1]])
device("DELL INSPIRON", [fs[0][0],
                         fs[1][1],
                         fs[2][1],
                         fs[3][1],
                         fs[4][0],
                         fs[5][1],
                         fs[6][2],
                         fs[7][2]])
device("IBUYPOWER GAMING", [fs[0][1],
                            fs[1][1],
                            fs[2][1],
                            fs[3][1],
                            fs[4][1],
                            fs[5][1],
                            fs[6][2],
                            fs[7][2]])
device("DELL XPS", [fs[0][0],
                    fs[1][1],
                    fs[2][1],
                    fs[3][1],
                    fs[4][1],
                    fs[5][2],
                    fs[6][2],
                    fs[7][2]])
device("LENOVO LEGION", [fs[0][1],
                         fs[1][1],
                         fs[2][1],
                         fs[3][1],
                         fs[4][2],
                         fs[5][2],
                         fs[6][2],
                         fs[7][2]])
device("MAC STUDIO", [fs[0][0],
                      fs[1][1],
                      fs[2][1],
                      fs[3][1],
                      fs[4][2],
                      fs[5][2],
                      fs[6][1],
                      fs[7][2]])
device("ARB 1", [fs[0][0],
                 fs[1][0],
                 fs[2][0],
                 fs[3][0],
                 fs[4][0],
                 fs[5][0],
                 fs[6][0],
                 fs[7][0]])
device("ARB 2", [fs[0][0],
                 fs[1][0],
                 fs[2][1],
                 fs[3][0],
                 fs[4][0],
                 fs[5][0],
                 fs[6][0],
                 fs[7][0]])
device("ARB 3", [fs[0][0],
                 fs[1][1],
                 fs[2][0],
                 fs[3][0],
                 fs[4][0],
                 fs[5][0],
                 fs[6][1],
                 fs[7][0]])
device("ARB 4", [fs[0][0],
                 fs[1][1],
                 fs[2][0],
                 fs[3][0],
                 fs[4][0],
                 fs[5][0],
                 fs[6][0],
                 fs[7][1]])
device("ARB 5", [fs[0][0],
                 fs[1][1],
                 fs[2][0],
                 fs[3][0],
                 fs[4][0],
                 fs[5][0],
                 fs[6][1],
                 fs[7][0]])
device("ARB 6", [fs[0][0],
                 fs[1][1],
                 fs[2][1],
                 fs[3][0],
                 fs[4][2],
                 fs[5][1],
                 fs[6][1],
                 fs[7][1]])
device("ARB 7", [fs[0][0],
                 fs[1][1],
                 fs[2][1],
                 fs[3][1],
                 fs[4][1],
                 fs[5][1],
                 fs[6][1],
                 fs[7][1]])
device("ARB 8", [fs[0][0],
                 fs[1][1],
                 fs[2][1],
                 fs[3][0],
                 fs[4][2],
                 fs[5][1],
                 fs[6][2],
                 fs[7][1]])
device("ARB 9", [fs[0][0],
                 fs[1][1],
                 fs[2][1],
                 fs[3][1],
                 fs[4][0],
                 fs[5][1],
                 fs[6][2],
                 fs[7][2]])
device("ARB 10", [fs[0][1],
                  fs[1][1],
                  fs[2][0],
                  fs[3][1],
                  fs[4][0],
                  fs[5][1],
                  fs[6][2],
                  fs[7][1]])
device("ARB 11", [fs[0][1],
                  fs[1][0],
                  fs[2][1],
                  fs[3][1],
                  fs[4][0],
                  fs[5][1],
                  fs[6][1],
                  fs[7][1]])
device("ARB 12", [fs[0][1],
                  fs[1][0],
                  fs[2][1],
                  fs[3][1],
                  fs[4][1],
                  fs[5][1],
                  fs[6][1],
                  fs[7][2]])
device("ARB 13", [fs[0][0],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][1],
                  fs[5][2],
                  fs[6][1],
                  fs[7][2]])
device("ARB 14", [fs[0][0],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][2],
                  fs[5][2],
                  fs[6][1],
                  fs[7][2]])
device("ARB 15", [fs[0][1],
                  fs[1][0],
                  fs[2][1],
                  fs[3][0],
                  fs[4][2],
                  fs[5][2],
                  fs[6][2],
                  fs[7][1]])
device("ARB 16", [fs[0][0],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][1],
                  fs[5][2],
                  fs[6][2],
                  fs[7][2]])
device("ARB 17", [fs[0][1],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][2],
                  fs[5][2],
                  fs[6][1],
                  fs[7][2]])
device("ARB 18", [fs[0][0],
                  fs[1][1],
                  fs[2][0],
                  fs[3][1],
                  fs[4][1],
                  fs[5][2],
                  fs[6][2],
                  fs[7][2]])
device("ARB 19", [fs[0][1],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][1],
                  fs[5][2],
                  fs[6][2],
                  fs[7][2]])
device("ARB 20", [fs[0][1],
                  fs[1][1],
                  fs[2][1],
                  fs[3][1],
                  fs[4][2],
                  fs[5][2],
                  fs[6][2],
                  fs[7][1]])


# %%
class feature_stance:
    ''' Class for importance and side on a given feature.'''
    count = 0
    feat_stances = []

    def __init__(self, featurename, side='pro', importance='A'):
        ''' Constructor for feature_stance().  If the featurename is not already a feature, 
        create a new feature.'''
        if not featurename.upper() in feature.features:
            feature(featurename)
        self.feature: feature = feature.features[featurename.upper()]
        self.side: str = side.upper()
        self.importance: str = importance.upper()
        
        self.count = feature_stance.count
        feature_stance.count += 1
        feature_stance.feat_stances.append(self)

    def __repr__(self):
        ''' Print out code that evaluates to this stance.'''
        return f'stance({self.feature.name}, {self.side}, {self.importance})'

    def __str__(self):
        ''' Return string version of self '''
        return f"<stance ({self.count}): {self.feature.name} [{self.side}:{self.importance}]>"

    def __eq__(self, other):
        ''' Overload == operator. Two stances must match issue and side, 
        though not importance. '''
        return self.feature == other.feature and self.side == other.side

# %%
class stance:
    ''' Class for importance and side on a given device.'''
    count = 0
    stances = []

    def __init__(self, device, featstances):
        ''' Constructor for stance().'''
        self.device: device = device.devices[device.upper()]
        self.side, self.importance = self.rate(featstances)
        self.side: str
        self.importance: str
        self.featstances: list[feature_stance] = featstances
        
        self.count = stance.count
        stance.count += 1
        stance.stances.append(self)

    def __repr__(self):
        ''' Print out code that evaluates to this stance.'''
        return f'stance({self.device.name}, {self.side}, {self.importance})'

    def __str__(self):
        ''' Return string version of self '''
        return f"<stance ({self.count}): {self.device.name} [{self.side}:{self.importance}]>"


# %% [markdown]
# #### Broad classes of customers considered  
#  - Gaming
#  - Schoolwork
#  - Web browsing/home desktop
#  - Professional editing

# %%
class agent:

    '''Class for agents who have goals.'''

    count = 0
    agents = []
    '''
    Fields: name, pronouns, goals, count, past, relations, age
    '''
    def __init__(self, name, pronouns='he him his', age=30):
        ''' Constructor for agent with name.'''
        self.name = name
        self.pronouns = pronouns
        self.count = agent.count
        agent.count += 1
        agent.agents.append(self)

        self.goals: list[feature_stance] = []
        self.past: list[stance] = []
        self.relations: list[stance] = []
        # Someone less familiar with computers will greater value relationships
        self.is_new: bool = False

    def __repr__(self):
        ''' Print out agent so that it can evaluate to itself.'''
        return f"agent({self.name!r})"

    def __str__(self):
        '''Return agent as a string.'''
        return f"<agent.name: {self.name} ({self.count})>"

    def add_purchase(self, stance):
        if stance not in self.past:
            self.past.append(stance)

    def add_relation(self, other_stances):
        for stnc in other_stances:
            self.relations.append(stnc)

    def add_goal(self, goal: feature_stance):
        for i, g in enumerate(self.goals):
            if g.feature == goal.feature:
                # Only replace the goal if there's a major change:
                # change in sides or increase in importance
                if g.side != goal.side or g.importance > goal.importance:
                    self.goals[i] = goal
                return
        self.goals.append(goal)

    def set_gamer(self, heavy=0):
        # The light gamer won't be running state-of-the-art AAA games, but still needs good enough 
        # performance to run games. Storage is needed to store more games, and memory might be a 
        # bottleneck for some games. As a light gamer, there is not really a need to go for the most
        # expensive option.
        light_gamer_goals = [feature_stance("LOW SINGLE-CORE PERFORMANCE", 'con', 'B'),
                             feature_stance("LOW GPU MEMORY", 'con', 'B'),
                             feature_stance("LUXURY", 'con', 'B'),
                             feature_stance("LOW STORAGE", 'con', 'B'),
                             feature_stance("LOW MEMORY", 'con', 'B')]
        # The heavy gamer wants to run state-of-the-art AAA games and/or competitive FPS games, 
        # both of which require not only CPU performance but very good GPU performance. The FPS player
        # in particular values 120 Hz refresh monitors. Both gamers fear storage that's too small 
        # to hold their games and/or highlights as well as memory becoming a bottleneck for performance. 
        # The AAA gamer in particular values good colors to better take in the experience.
        heavy_gamer_goals = [feature_stance("120+ HZ REFRESH", 'pro', 'A'),
                             feature_stance("HIGH BRIGHTNESS/COLOR QUALITY", 'pro', 'B'),
                       feature_stance("MEDIUM-HIGH SINGLE CORE PERFORMANCE", 'pro', 'B'),
                       feature_stance("HIGH GPU MEMORY", 'pro', 'A'),
                       feature_stance("LOW STORAGE", 'con', 'B'),
                       feature_stance("LOW MEMORY", 'con', 'A')]
        if heavy:
            for g in heavy_gamer_goals:
                self.add_goal(g)
        else:
            for g in light_gamer_goals:
                self.add_goal(g)

    def set_student(self, research=0):
        # Multi-tasking is extremely valuable for schoolwork

        # For a regular student, single-core performance could bottleneck the occasional heavy software
        # a class might need, more cores allows for a much better multi-tasking experience, budget is
        # certainly a bottleneck, and low storage could lead to excessive need of deleting files that
        # may be useful in the future.
        schoolwork_goals = [feature_stance("LOW SINGLE-CORE PERFORMANCE", 'con', 'C'),
                            feature_stance("6+ CORES", 'pro', 'B'),
                            feature_stance("BUDGET", 'pro', 'A'),
                            feature_stance("MODERATE PRICE", 'pro', 'C'),
                            feature_stance("LOW STORAGE", 'con', 'C')]
        # For a heavier researcher (e.g. someone writing a senior thesis who "needs" 40+ tabs open), 
        # multi-tasking becomes even more important, price could become less important (but still a 
        # valid concern), running out of storage becomes a very real worry, and memory becomes a 
        # concern (thanks, Chrome).
        researcher_goals = [feature_stance("MEDIUM-HIGH SINGLE CORE PERFORMANCE", 'pro', 'B'),
                            feature_stance("6+ CORES", 'pro', 'A'),
                            feature_stance("LUXURY", 'con', 'B'),
                            feature_stance("LOW STORAGE", 'con', 'A'),
                            feature_stance("LOW MEMORY", 'con', 'A')]
        if research:
            for g in researcher_goals:
                self.add_goal(g)
        else:
            for g in schoolwork_goals:
                self.add_goal(g)
        
    def set_home(self, media=0):
        # This category essentially consists of "casual" users, who want to browse the web and consume media,
        # maybe do their taxes or use other relatively light software.

        # For a regular home desktop that's primarily intended for web browsing, there's not much needed. 
        # Higher CPU performance can be nice, as can higher storage and memory, but other than that, the 
        # main factor is the budget.
        desktop_goals = [feature_stance("MEDIUM-HIGH SINGLE CORE PERFORMANCE", 'pro', 'C'),
                         feature_stance("BUDGET", 'PRO', 'A'),
                         feature_stance("MODERATE PRICE", 'PRO', 'C'),
                         feature_stance("LOW STORAGE", 'con', 'C'),
                         feature_stance("LOW MEMORY", "con", 'C')]
        # If the computer is intended for frequent use for watching movies and the like, more color quality
        # and potentially a higher refresh rate could be desirable. Personally, I can only tell the 
        # difference between 60 Hz and 120 Hz when I play games, but some people say they can feel the 
        # difference when watching media.
        media_goals = [feature_stance("HIGH BRIGHTNESS/COLOR QUALITY", 'pro', 'B'),
                       feature_stance("MEDIUM-HIGH SINGLE CORE PERFORMANCE", 'pro', 'C'),
                        feature_stance("BUDGET", 'PRO', 'A'),
                        feature_stance("MODERATE PRICE", 'PRO', 'C'),
                        feature_stance("LOW STORAGE", 'con', 'C'),
                        feature_stance("LOW MEMORY", "con", 'C')]
        if media:
            for g in media_goals:
                self.add_goal(g)
        else:
            self.is_new = True
            for g in desktop_goals:
                self.add_goal(g)
    
    def set_editor(self):
        # A media editor definitely needs a beefy computer in terms of most considerations. 
        # CPU and GPU performance work in tandem to improve the workflow; multi-tasking, storage,
        # and memory are all important for performance; and color quality could allow for more accurate products.
        editing_goals = [feature_stance("HIGH BRIGHTNESS/COLOR QUALITY", 'pro', 'B'),
                         feature_stance("MEDIUM-HIGH SINGLE CORE PERFORMANCE", 'pro', 'B'),
                         feature_stance("6+ CORES", 'pro', 'B'),
                         feature_stance("HIGH GPU MEMORY", 'pro', 'A'),
                         feature_stance("LOW GPU MEMORY", 'con', 'A'),
                         feature_stance("HIGH STORAGE", 'pro', 'A'),
                         feature_stance("LOW STORAGE", 'con', 'A'),
                         feature_stance("HIGH MEMORY", 'pro', 'B'),
                         feature_stance("LOW MEMORY", 'con', 'A')]
        for g in editing_goals:
            self.add_goal(g)

    def pp(self):
        '''Pretty print agent information.'''
        result = f"Name:\t{self.name}"
        if self.goals:
            result += f"\nGoals:\t{self.goals}"
        if self.pronouns:
            result += f"\nPronouns:\t{self.pronouns}"
        return result

    def __eq__(self, other):
        ''' Overload == operator.  Are two agents equal by name and goals? '''
        return self.name == other.name and sorted(self.goals) == sorted(other.goals)

    def copy(self):
        ''' Clone the agent, including name, and goals. '''
        newagent = agent(self.name, age=self.age)
        newagent.goals = self.goals[:]
        return newagent


# %%
# The most basic valuation of a device: weigh pros and cons
# exhibited by the device.
def get_val(fs: feature_stance) -> tuple[int, int]:
    if fs.side == 'PRO':
        if fs.importance == 'A':
            return 3, 0
        elif fs.importance == 'B':
            return 2, 0
        else:
            return 1, 0
    else:
        assert(fs.side == 'CON')
        if fs.importance == 'A':
            return 0, 3
        elif fs.importance == 'B':
            return 0, 2
        else:
            return 0, 1
def add(x: tuple[int, int], y: tuple[int, int], flip=False) -> tuple[int, int]:
    return (x[0]+y[0], x[1]+y[1]) if not flip else (x[0]+y[1], x[1]+y[0])

# %% [markdown]
# #### On decision strategies used
#  1. The agent has preferences for features encoded as feature_stances in their goals field. A weighted sum of the pros and cons of the device as they pertain to the agent's goals is the first filter: if the pros vastly outweight the cons, then the device should be good for the agent and vice versa.  
#  **Note:** a lack of a pro is a con, and a lack of a con is a pro.
#  2. People usually consider a single strong negative to outweight several moderate positives. I incorporate this reasoning by searching for a "deal breaker" next -- a single strong reason to not recommend the device.
#  3. At this point, there are no glaringly obvious, specific reasons to recommend or not recommend the device. Thus, I consider holistic evaluations next. The most relevant one would be past purchases, whether it's by the agent themselves or by someone they have a relation to. Whoever made that purchase would have an ex-post evaluation of the overall device, and if it's a strong evaluation in either direction, then that should strongly influence the decision to recommend or not. If there's mixed reviews, further detailed consideration is needed.
#  4. Finally, if there are at least two strong reasons to buy the computer feature-wise, the device is recommended. Otherwise, ignore C-level (unimportant) goals and redo calculations in step 1; recommend the device if and only if there is not a strong reason to not recommend the device (i.e. if and only if the cons don't vastly outweight the pros).
# #### On incorporating VOTE
#   - **Preferences:** The primary driving factor of the decision to recommend or not is precisely the preferences of the agent, encoded in their goals.
#   - **History of prior decisions:** Prior decisions and their results factor into the decision at step 3.
#   - **Relationships:** What makes a negative relationship in the context of a computer recommendation? A computer either worked well for the purposes of the individual who bought it or not. A decision of positive vs. negative relationships with someone else in terms of whether their goals align or not (i.e. whether or not their preferences for device features are the same or not) are already encoded within the individual's goals, too. Thus, the main purpose of relationships in the context of computer recommendations is whether or not the device as a bundle functions well together, i.e. it has a cohesive set of features. Therefore, I incorporated the relationships aspect of VOTE in the history of prior decisions aspect -- a past purchase of someone related factors into the decision similarly to a prior decision of the agent themselves.
# #### On my implementation
# My understanding is that if, for example, a salesperson were to go to a customer to help them decide what computer to buy (or whether or not to buy a computer), they would essentially be gathering the customer's goals. For example, if the customer said they value quality images or that they like watching cinematic movies, then the salesperson would add "120+ Hz monitor" and "high-quality color/brightness" as pros in their mental model of the customer. In a similar vein, I'd expect other factors like relationships with other people (What computers do the people around you have? What kind of computer does your employer prefer?), compatibility (What other devices do you use? Do you prefer Windows to Mac or Linux? Do you want to play games that are only available on Windows or use a software that's best integrated with MacOS?), etc. to also be converted into corresponding goals within the salesperson's mental model of the customer.  
# To this extent, I've implemented "likes" to be the salesperson's interpretation of whether or not the customer would like a device given sufficient interrogation. That is, I note that the VOTE models does not distinguish between goals that the agent knows they want versus goals that they'd like to accomplish but they're unaware that such an option exists. I then consider an agent's goals to be the entire set of goals that, if accomplished, they'd be happy with, regardless of whether or not they're aware of it prior. Under this interpretation, the bulk of the work is in "likes," with "prefers" and "recommend" simply wrapping "likes" within another function to judge several or all devices respectively. I hope that this interpretation is what was intended, but if it wasn't, I at least wanted to explain what I thought we were being asked to do.

# %%
def pronoun(agent, case):
    pronouns = agent.pronouns.split()
    cases = {'subj': 0, 'obj': 1, 'poss': 2}
    return pronouns[cases[case]]

# %%
# Outputs yes/no, strength (0/1/2 for A/B/C), and string for reason
def likes(agent: agent, device: device) -> tuple[bool, int, str]:
    val = (0, 0)
    goal_vals = []
    reason = []
    # Step 1
    goal_sat = [[[0, 0] for j in range(2)] for i in range(3)]
    for goal in agent.goals:
        goal_treated = False
        for feat in device.features:
            if goal.feature == feat:
                goal_vals.append((get_val(goal),False))
                val = add(val, goal_vals[-1][0])
                goal_treated = True
                goal_sat[ord(goal.importance)-ord('A')][goal.side == 'PRO'][0] += 1
        # The lack of a pro is bad, the lack of a con is good
        if not goal_treated:
            goal_vals.append((get_val(goal),True))
            val = add(val, goal_vals[-1][0], True)
            goal_sat[ord(goal.importance)-ord('A')][goal.side == 'PRO'][1] += 1
    # If the pros vastly outweigh the cons, then the flaws the device
    # might have can probably be overlooked.
    if val[0] >= 2 * val[1]:
        reason.append(' '.join(["Strongly recommend", device.name, "to", agent.name]))
        reason.append("\tPros vastly outweigh cons for " + pronoun(agent, 'obj') + ":")
        reason.append(' '.join(["\t\tNumber of pros satisfied:",str(goal_sat[0][1][0]),"importance A",
                                str(goal_sat[1][1][0]),"importance B",str(goal_sat[2][1][0]),"importance C"]))
        reason.append(' '.join(["\t\tNumber of pros not satisfied:",str(goal_sat[0][1][1]),"importance A",
                                str(goal_sat[1][1][1]),"importance B",str(goal_sat[2][1][1]),"importance C"]))
        reason.append(' '.join(["\t\tNumber of cons satisfied:",str(goal_sat[0][0][0]),"importance A",
                                str(goal_sat[1][0][0]),"importance B",str(goal_sat[2][0][0]),"importance C"]))
        reason.append(' '.join(["\t\tNumber of cons not satisfied:",str(goal_sat[0][0][1]),"importance A",
                                str(goal_sat[1][0][1]),"importance B",str(goal_sat[2][0][1]),"importance C"]))
        return True, 0, '\n'.join(reason)
    
    # Step 2
    # Generally, people consider a single strong negative to outweigh
    # several moderate positives: here I look for a "deal-breaker."
    for i, v in enumerate(goal_vals):
        if v[0] == (0, 3) and not v[1] or v[0] == (3, 0) and v[1]:
            reason.append(' '.join(["Strongly do not recommend", device.name, "to", agent.name]))
            reason.append("\tDeal-breaker detected for "+pronoun(agent, 'obj')+":")
            if agent.goals[i].side == 'PRO':
                reason.append(' '.join(["\t\tFeature", agent.goals[i].feature.name, "desired at importance level",
                                         agent.goals[i].importance, "but it's not in the device"]))
            else:
                reason.append(' '.join(["\t\tFeature", agent.goals[i].feature.name, "not wanted at priority level",
                                         agent.goals[i].importance, "but it's in device"]))
            return False, 0, '\n'.join(reason)
        
    # Step 3
    # Now we consider past purchases and, if relevant to this agent and device,
    # purchases of related people
    past_purchases = agent.past
    if agent.is_new:
        past_purchases += agent.relationships()
    relevant = []
    for purch in past_purchases:
        if purch.device == device: relevant.append(purch)
    if len(relevant)>1:
        # Compare strong stances
        count = 0,0
        for purch in past_purchases:
            if purch.importance == 'A':
                count = add(count, (1,0) if purch.side == 'PRO' else (0,1), False)
        if count[0] > count[1]:
            reason.append(' '.join(["Moderately recommend", device.name, "to", agent.name]))
            reason.append("\tPrevious purchases of",device.name,"were quite good")
            return True, 1, '\n'.join(reason)
        elif count[1] > count[0]:
            reason.append(' '.join(["Moderately do not recommend", device.name, "to", agent.name]))
            reason.append("\tPrevious purchases of",device.name,"were quite bad")
            return False, 1, '\n'.join(reason)
    elif len(relevant)==1:
        if relevant[0].importance < 'C':
            if relevant[0].side == 'PRO':
                reason.append(' '.join(["Moderately recommend", device.name, "to", agent.name]))
                reason.append("\tPrevious purchase of",device.name,"was quite good")
            else:
                reason.append(' '.join(["Moderately do not recommend", device.name, "to", agent.name]))
                reason.append("\tPrevious purchase of",device.name,"was quite bad")
            return relevant[0].side == 'PRO', 1, '\n'.join(reason)
        
    # Step 4
    # Two strong reasons to buy the computers is sufficient
    if goal_sat[0][1][0]+goal_sat[0][0][1] >= 2:
        reason.append(' '.join(["Slightly recommend", device.name, "to", agent.name]))
        reason.append("\tThere are strong reasons for "+pronoun(agent, 'obj')+" to buy the computer despite any downsides:")
        return True, 2, '\n'.join(reason)

    # Otherwise, recalc valuation without 'C' goals. If pros are at least the 3/4 the cons, recommend.
    val = (0, 0)
    goal_vals = []
    for goal in agent.goals:
        if goal.importance < 'C':
            goal_treated = False
            for feat in device.features:
                if goal.feature == feat:
                    goal_vals.append((get_val(goal),False))
                    val = add(val, goal_vals[-1][0])
                    goal_treated = True
            # The lack of a pro is bad, the lack of a con is good
            if not goal_treated:
                goal_vals.append((get_val(goal),True))
                val = add(val, goal_vals[-1][0], True)
    if val[0] >= 3/4 * val[1]:
        reason.append(' '.join(["Slightly recommend",device.name, "to", agent.name]))
        reason.append("\tThere are enough pros to this device for "+pronoun(agent, 'obj')+" to consider, but nothing majorly convincing.")
        return True, 2, '\n'.join(reason)
    else:
        reason.append(' '.join(["A soft recommendation for",agent.name,"to not buy",device.name]))
        reason.append("\tThere are some upsides to this device for "+pronoun(agent, 'obj')+", but nothing impressively so")
        return False, 2, '\n'.join(reason)


bob = agent("Bob", "he him his")
bob.set_gamer(heavy=0)
bob.add_goal(feature_stance(featurename='MODERATE PRICE', side='pro', importance='A'))
print(bob.pp())
print(likes(bob, device.devices['ARB 10'])[2])

sam = agent("Sam", "they them their")
sam.set_home(media=1)
sam.add_goal(feature_stance(featurename="MEDIUM-HIGH SINGLE CORE PERFORMANCE", side='pro', importance='b'))
sam.add_goal(feature_stance(featurename="LOW MEMORY", side='con', importance='c'))
print(likes(sam, device.devices['HP DESKTOP'])[2])


# %%
def prefers(agent: agent, devices: list[device]):
    ret = []
    for d in devices:
        x = likes(agent, d)
        if x[0]:
            ret.append((d, x))
    ret.sort(key=lambda x: x[1][1])
    if len(ret) > 3:
        ret = ret[:3]
    if len(ret) == 0:
        print("Could not find a suitable device to recommend to", agent.name,"from given list")
    elif len(ret) == 1:
        print("Found 1 recommendation for", agent.name, "from given list of devices:")
    elif len(ret) == 2:
        print("Found 2 recommendations for", agent.name, "from given list of devices:")
    else:
        print("Top 3 recommendations for", agent.name, "from given list of devices:")
    for d, x in ret:
        print(x[2])
    return ret
    

devs = []
for i, d in enumerate(device.devices.keys()):
    if i % 3 == 0:
        devs.append(device.devices[d])
prefers(sam, devs)


# %%
def recommend(agent: agent):
    all_devices = []
    for d in device.devices.keys():
        all_devices.append(device.devices[d])
    ret = prefers(agent, all_devices)
    return ret
recommend(bob)
recommend(sam)



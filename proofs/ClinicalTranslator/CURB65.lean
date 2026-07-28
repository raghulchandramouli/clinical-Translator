namespace ClinicalTranslator

structure Facts where
  confusion : Bool
  elevatedUrea : Bool
  highRespiratoryRate : Bool
  lowBloodPressure : Bool
  ageAtLeast65 : Bool

def point (value : Bool) : Nat := if value then 1 else 0

def score (facts : Facts) : Nat :=
  point facts.confusion
    + point facts.elevatedUrea
    + point facts.highRespiratoryRate
    + point facts.lowBloodPressure
    + point facts.ageAtLeast65

def specification (facts : Facts) : Nat :=
  [facts.confusion, facts.elevatedUrea, facts.highRespiratoryRate,
    facts.lowBloodPressure, facts.ageAtLeast65].count true

def incompleteScore (facts : Facts) : Nat :=
  point facts.elevatedUrea
    + point facts.highRespiratoryRate
    + point facts.lowBloodPressure
    + point facts.ageAtLeast65

def confusionOnly : Facts where
  confusion := true
  elevatedUrea := false
  highRespiratoryRate := false
  lowBloodPressure := false
  ageAtLeast65 := false

theorem score_correct (facts : Facts) : score facts = specification facts := by
  cases facts with
  | mk confusion urea respiration pressure age =>
    cases confusion <;> cases urea <;> cases respiration <;>
      cases pressure <;> cases age <;> decide

theorem incomplete_counterexample :
    incompleteScore confusionOnly ≠ specification confusionOnly := by
  decide

theorem corrected_representation_sufficient (facts : Facts) :
    score facts = specification facts :=
  score_correct facts

def booleans := [false, true]

def allFacts : List Facts :=
  booleans.flatMap fun confusion =>
    booleans.flatMap fun elevatedUrea =>
      booleans.flatMap fun highRespiratoryRate =>
        booleans.flatMap fun lowBloodPressure =>
          booleans.map fun ageAtLeast65 =>
            { confusion, elevatedUrea, highRespiratoryRate,
              lowBloodPressure, ageAtLeast65 }

#eval allFacts.map score

end ClinicalTranslator

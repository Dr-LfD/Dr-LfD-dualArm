(define (problem skill_seg_problem)
  (:domain skill_seg)

  (:objects
    r1 r2 - robot
    obj1 obj2 drawer - object
    loc1 loc2 - surface
    skPick skPlace skPush skInsertion skHandOff - skill
  )

  (:init
    ; Object properties
    (LeftArm r1)
    (RightArm r2)
    ; (Region loc1)
    (SkillPick skPick)  
    (SkillPlace skPlace) 
    (SkillBiLR2LR skInsertion) 
    (SkillBiR2L skHandOff) 
    (SkillNudge skPush)

    ; Initial state of the world
    (ArmEmpty r1)
    (On obj1 loc1)
    (ArmEmpty r2)
    (On obj2 loc1)
  )

  (:goal
    (and
      ; ; pick place x2
      ; (DoneSkillPick skPick r1 obj1)
      ; (DoneSkillPlace skPlace r1 obj1 loc2)
      ; (DoneSkillPick skPick r1 obj2)
      ; (DoneSkillPlace skPlace r1 obj2 loc2)

      ; ; ; LR2LR
      ; (DoneSkillBiLR2LR skInsertion r1 r2 obj1 obj2)

      ; ; ; R2L
      ; (DoneSkillBiR2L skHandOff r1 r2  obj2)

      ;; push the drawer and pick the obj2 
      (DoneSkillNudge skPush r1 drawer)
      (DoneSkillPick skPick r2 obj2)

    )
  )
)

; (define (problem skill_seg_problem)
;   (:domain skill_seg)

;   (:objects
;     r1 r2 - robot
;     obj1 obj2 drawer - object
;     loc1 loc2 - surface
;     skPick_r1_obj1 skPlace_r1_obj1_loc2 skPick_r1_obj2  skPlace_r1_obj2_loc2 skPick_r2_obj2 skPush skInsertion_obj1_obj2 skHandOff_obj2 - skill
;   )

;   (:init
;     ; Object properties
;     (LeftArm r1)
;     (RightArm r2)
;     ; (Region loc1)

;     ; (TgtObj skPick_r1_obj1 r1 obj1)
;     ; (TgtObj skPlace_r1_obj1_loc2 r1 loc2)
;     ; (TgtObj skPick_r1_obj2 r1 obj2)
;     ; (TgtObj skPlace_r1_obj2_loc2 r1 loc2)
    
;     ; (TgtObj skPick_r1_obj1 r1 obj1)
;     ; (TgtObj skPick_r2_obj2 r2 obj2)
;     ; (TgtObj skInsertion_obj1_obj2 r1 obj1)
;     ; (TgtObj skInsertion_obj1_obj2 r2 obj2)

;     (TgtObj skPick_r2_obj2 r2 obj2)
;     (TgtObj skHandOff_obj2 r1 obj1)
;     (TgtObj skHandOff_obj2 r2 obj1)

;     ; Initial state of the world
;     (ArmEmpty r1)
;     (On obj1 loc1)
;     (ArmEmpty r2)
;     (On obj2 loc1)
;   )

;   (:goal
;     (and
;     ;   ; pick place x2
;     ; (DoneSkill skPick_r1_obj1 r1 obj1)
;     ; (DoneSkill skPlace_r1_obj1_loc2 r1 loc2)
;     ; (DoneSkill skPick_r1_obj2 r1 obj2)
;     ; (DoneSkill skPlace_r1_obj2_loc2 r1 loc2)

;       ; ; LR2LR
;     ; (DoneSkill skPick_r1_obj1 r1 obj1)
;     ; (DoneSkill skPick_r2_obj2 r2 obj2)
;     ;   (DoneSkill skInsertion_obj1_obj2 r1  obj1)
;     ;   (DoneSkill skInsertion_obj1_obj2 r2  obj2)

;       ; ; R2L
;     (DoneSkill skPick_r2_obj2 r2 obj2)
;     (DoneSkill skHandOff_obj2 r1 obj1)
;     (DoneSkill skHandOff_obj2 r2 obj1)

;       ; ;; push the drawer and pick the obj2 
;       ; (DoneSkillNudge skPush r1 drawer)
;       ; (DoneSkillPick skPick r2 obj2)

;     )
;   )
; )
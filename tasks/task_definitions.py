from typing import Dict, Any, List, Tuple
from src.models import Action
import random


def strict_unit_score(score: float, eps: float = 1e-3) -> float:
    """Clamp score to strict open interval (0, 1)."""
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


class TaskDefinition:
    """Base class for task definitions"""
    
    def __init__(self, task_id: str, difficulty: str):
        self.task_id = task_id
        self.difficulty = difficulty
    
    def generate_request(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a test request and its ground truth"""
        raise NotImplementedError
    
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """Grade the agent's action. Returns (score, feedback, breakdown)"""
        raise NotImplementedError


class EasyTask(TaskDefinition):
    """
    Easy: Simple Schema Validation
    Agent validates a user registration request against a fixed schema.
    """
    
    def __init__(self):
        super().__init__("easy_validation", "easy")
        self.endpoint = "user_registration"
        self.schema = {
            "type": "object",
            "required": ["username", "email", "age"],
            "properties": {
                "username": {"type": "string", "minLength": 3},
                "email": {"type": "string", "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                "age": {"type": "integer", "minimum": 13, "maximum": 120}
            }
        }
    
    def generate_request(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate valid or invalid registration requests"""
        scenarios = [
            # Valid request
            {
                "payload": {
                    "username": "alice123",
                    "email": "alice@example.com",
                    "age": 25
                },
                "is_valid": True,
                "correct_endpoint": "user_registration",
                "issues": []
            },
            # Missing required field
            {
                "payload": {
                    "username": "bob456",
                    "email": "bob@example.com"
                },
                "is_valid": False,
                "correct_endpoint": "user_registration",
                "issues": ["missing_required_field: age"]
            },
            # Invalid type
            {
                "payload": {
                    "username": "charlie",
                    "email": "charlie@example.com",
                    "age": "twenty-five"
                },
                "is_valid": False,
                "correct_endpoint": "user_registration",
                "issues": ["invalid_type: age (expected integer)"]
            },
            # Invalid email format
            {
                "payload": {
                    "username": "dave",
                    "email": "not-an-email",
                    "age": 30
                },
                "is_valid": False,
                "correct_endpoint": "user_registration",
                "issues": ["invalid_format: email"]
            },
            # Age out of range
            {
                "payload": {
                    "username": "eve",
                    "email": "eve@example.com",
                    "age": 10
                },
                "is_valid": False,
                "correct_endpoint": "user_registration",
                "issues": ["invalid_value: age (below minimum 13)"]
            },
        ]
        
        scenario = random.choice(scenarios)
        return scenario["payload"], scenario
    
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """Grade the validation action"""
        breakdown = {}
        
        # Check if action type is appropriate
        if ground_truth["is_valid"]:
            # Valid request should be routed
            if action.action_type == "route" and action.target_endpoint == ground_truth["correct_endpoint"]:
                breakdown["correct_action"] = 0.5
                breakdown["correct_routing"] = 0.5
                return strict_unit_score(1.0), "Perfect! Valid request correctly routed.", breakdown
            elif action.action_type == "validate":
                breakdown["correct_action"] = 0.3
                return strict_unit_score(0.3), "Request is valid but not routed yet.", breakdown
            else:
                breakdown["incorrect_action"] = 0.0
                return strict_unit_score(0.0), "Valid request should be routed, not rejected/fixed.", breakdown
        else:
            # Invalid request should be rejected or identified for fixing
            if action.action_type == "reject":
                # Check if issues were identified
                if action.validation_issues:
                    issues_found = len(set(action.validation_issues) & set(ground_truth["issues"]))
                    total_issues = len(ground_truth["issues"])
                    issue_score = issues_found / total_issues if total_issues > 0 else 0
                    
                    breakdown["correct_action"] = 0.5
                    breakdown["issue_identification"] = 0.5 * issue_score
                    
                    total = 0.5 + 0.5 * issue_score
                    feedback = f"Correctly rejected. Identified {issues_found}/{total_issues} issues."
                    return strict_unit_score(total), feedback, breakdown
                else:
                    breakdown["correct_action"] = 0.5
                    return strict_unit_score(0.5), "Correctly rejected but no issues identified.", breakdown
            elif action.action_type == "fix":
                breakdown["attempted_fix"] = 0.3
                return strict_unit_score(0.3), "Good attempt at fixing, but rejection is safer for this task.", breakdown
            else:
                breakdown["incorrect_action"] = 0.0
                return strict_unit_score(0.0), "Invalid request should be rejected or fixed.", breakdown
class MediumTask(TaskDefinition):
    """
    Medium: Multi-Endpoint Routing
    Agent must identify correct endpoint from multiple options and validate.
    """
    
    def __init__(self):
        super().__init__("medium_routing", "medium")
        self.endpoints = {
            "user_registration": {
                "required": ["username", "email", "age"],
                "optional": ["phone"]
            },
            "product_create": {
                "required": ["name", "price", "category"],
                "optional": ["description", "stock"]
            },
            "order_submit": {
                "required": ["user_id", "items", "total"],
                "optional": ["shipping_address", "coupon"]
            },
            "payment_process": {
                "required": ["order_id", "amount", "method"],
                "optional": ["card_last4"]
            },
            "support_ticket": {
                "required": ["user_id", "subject", "priority"],
                "optional": ["description", "attachment"]
            }
        }
    
    def generate_request(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate requests for different endpoints"""
        scenarios = [
            {
                "payload": {
                    "name": "Laptop",
                    "price": 999.99,
                    "category": "Electronics",
                    "description": "High-performance laptop"
                },
                "correct_endpoint": "product_create",
                "missing_fields": [],
                "is_valid": True
            },
            {
                "payload": {
                    "user_id": "user_123",
                    "items": [{"id": "prod_1", "qty": 2}],
                    "total": 50.00
                },
                "correct_endpoint": "order_submit",
                "missing_fields": [],
                "is_valid": True
            },
            {
                "payload": {
                    "order_id": "ord_456",
                    "amount": 50.00,
                    "method": "credit_card"
                },
                "correct_endpoint": "payment_process",
                "missing_fields": [],
                "is_valid": True
            },
            {
                "payload": {
                    "user_id": "user_789",
                    "subject": "Login issue",
                    "priority": "high"
                },
                "correct_endpoint": "support_ticket",
                "missing_fields": [],
                "is_valid": True
            },
            # Ambiguous - could be multiple endpoints
            {
                "payload": {
                    "user_id": "user_999",
                    "email": "test@example.com"
                },
                "correct_endpoint": "user_registration",  # Most likely
                "missing_fields": ["username", "age"],
                "is_valid": False
            },
        ]
        
        scenario = random.choice(scenarios)
        return scenario["payload"], scenario
    
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """Grade routing and validation"""
        breakdown = {}
        
        # Check endpoint routing
        if action.target_endpoint == ground_truth["correct_endpoint"]:
            breakdown["correct_routing"] = 0.6
            routing_feedback = "Correct endpoint identified."
        else:
            breakdown["correct_routing"] = 0.0
            routing_feedback = f"Wrong endpoint. Expected {ground_truth['correct_endpoint']}."
        
        # Check validation
        if ground_truth["is_valid"]:
            if action.action_type == "route":
                breakdown["correct_validation"] = 0.4
                validation_feedback = "Correctly validated as complete."
            else:
                breakdown["correct_validation"] = 0.0
                validation_feedback = "Valid request should be routed."
        else:
            if action.action_type in ["reject", "fix"]:
                # Check if missing fields were identified
                if action.validation_issues:
                    identified = any(field in str(action.validation_issues) 
                                   for field in ground_truth["missing_fields"])
                    if identified:
                        breakdown["correct_validation"] = 0.4
                        validation_feedback = "Correctly identified missing fields."
                    else:
                        breakdown["correct_validation"] = 0.2
                        validation_feedback = "Identified as invalid but missed specific fields."
                else:
                    breakdown["correct_validation"] = 0.1
                    validation_feedback = "Identified as invalid but no details."
            else:
                breakdown["correct_validation"] = 0.0
                validation_feedback = "Invalid request should be rejected/fixed."
        
        total_score = sum(breakdown.values())
        feedback = f"{routing_feedback} {validation_feedback}"
        
        return strict_unit_score(total_score), feedback, breakdown
class HardTask(TaskDefinition):
    """
    Hard: Malformed Request Recovery
    Agent must attempt to fix ambiguous or corrupted requests.
    """
    
    def __init__(self):
        super().__init__("hard_recovery", "hard")
    
    def generate_request(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate malformed requests with possible fixes"""
        scenarios = [
            # Type coercion needed
            {
                "payload": {
                    "username": "alice",
                    "email": "alice@example.com",
                    "age": "25"  # String instead of int
                },
                "correct_endpoint": "user_registration",
                "fixable": True,
                "expected_fix": {
                    "age": 25
                },
                "issue": "type_coercion_needed"
            },
            # Field name typo
            {
                "payload": {
                    "user_name": "bob",  # Should be 'username'
                    "email": "bob@example.com",
                    "age": 30
                },
                "correct_endpoint": "user_registration",
                "fixable": True,
                "expected_fix": {
                    "username": "bob"
                },
                "issue": "field_name_typo"
            },
            # Extra/unknown fields
            {
                "payload": {
                    "product_name": "Laptop",  # Should be 'name'
                    "price": "999.99",  # Should be float
                    "category": "Electronics",
                    "random_field": "ignore_me"
                },
                "correct_endpoint": "product_create",
                "fixable": True,
                "expected_fix": {
                    "name": "Laptop",
                    "price": 999.99
                },
                "issue": "field_mapping_and_type_coercion"
            },
            # Nested structure flattening needed
            {
                "payload": {
                    "order": {
                        "id": "ord_123",
                        "amount": 100.00,
                        "payment_method": "credit_card"
                    }
                },
                "correct_endpoint": "payment_process",
                "fixable": True,
                "expected_fix": {
                    "order_id": "ord_123",
                    "amount": 100.00,
                    "method": "credit_card"
                },
                "issue": "nested_structure_and_field_mapping"
            },
            # Completely corrupt - not fixable
            {
                "payload": {
                    "xyz": 123,
                    "abc": "test"
                },
                "correct_endpoint": None,
                "fixable": False,
                "expected_fix": None,
                "issue": "corrupt_beyond_recovery"
            }
        ]
        
        scenario = random.choice(scenarios)
        return scenario["payload"], scenario
    
    def grade(self, action: Action, ground_truth: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """Grade recovery attempt"""
        breakdown = {}
        
        if not ground_truth["fixable"]:
            # Should reject corrupt requests
            if action.action_type == "reject":
                breakdown["correct_action"] = 1.0
                return strict_unit_score(1.0), "Correctly rejected unfixable request.", breakdown
            elif action.action_type == "fix":
                breakdown["attempted_impossible_fix"] = 0.2
                return strict_unit_score(0.2), "Attempted fix on unfixable request, but good intent.", breakdown
            else:
                breakdown["incorrect_action"] = 0.0
                return strict_unit_score(0.0), "Corrupt request should be rejected.", breakdown
        
        # Fixable request
        if action.action_type == "fix":
            breakdown["attempted_fix"] = 0.4
            
            # Check if fix suggestion is provided
            if action.fix_suggestion:
                # Check endpoint identification
                if action.target_endpoint == ground_truth["correct_endpoint"]:
                    breakdown["correct_endpoint"] = 0.3
                else:
                    breakdown["correct_endpoint"] = 0.0
                
                # Check if fix is correct
                expected_fix = ground_truth["expected_fix"]
                suggested_fix = action.fix_suggestion
                
                # Simple matching - check if expected keys/values are in suggestion
                matches = 0
                for key, value in expected_fix.items():
                    if key in suggested_fix and suggested_fix[key] == value:
                        matches += 1
                
                fix_quality = matches / len(expected_fix) if expected_fix else 0
                breakdown["fix_quality"] = 0.3 * fix_quality
                
                total = sum(breakdown.values())
                feedback = f"Fix attempted. Quality: {fix_quality:.1%}. Endpoint: {action.target_endpoint}"
                return strict_unit_score(total), feedback, breakdown
            else:
                feedback = "Fix attempted but no suggestion provided."
                return strict_unit_score(0.4), feedback, breakdown
        
        elif action.action_type == "reject":
            breakdown["safe_choice"] = 0.5
            return strict_unit_score(0.5), "Rejected fixable request - safe but suboptimal.", breakdown
        
        else:
            breakdown["incorrect_action"] = 0.0
            return strict_unit_score(0.0), "Malformed request should be fixed or rejected.", breakdown


# Task registry
TASKS = {
    "easy": EasyTask(),
    "medium": MediumTask(),
    "hard": HardTask()
}
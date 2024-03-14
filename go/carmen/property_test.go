package carmen

import "testing"

func TestProperty_CannotParse(t *testing.T) {
	properties := make(Properties)
	properties["x"] = "not a number"

	if _, err := properties.GetInteger("x", 10); err == nil {
		t.Errorf("parsing property should fail")
	}
}

func TestProperty_GetDefaultValue(t *testing.T) {
	properties := make(Properties)

	if val, err := properties.GetInteger("y", 10); err != nil || val != 10 {
		t.Errorf("cannot get default value: wanted: %d, got: %d, %v", 10, val, err)
	}
}

func TestProperty_GetValue(t *testing.T) {
	properties := make(Properties)
	properties["y"] = "10"

	if val, err := properties.GetInteger("y", 10); err != nil || val != 10 {
		t.Errorf("cannot get default value: wanted: %d, got: %d, %v", 10, val, err)
	}
}

func TestProperty_SetValue(t *testing.T) {
	properties := make(Properties)

	properties.SetInteger("y", 10)
	if val, err := properties.GetInteger("y", 10); err != nil || val != 10 {
		t.Errorf("cannot get default value: wanted: %d, got: %d, %v", 10, val, err)
	}
}

func TestProperty_CanInitProperties(t *testing.T) {
	var properties Properties

	properties.SetInteger("y", 10)
	if val, err := properties.GetInteger("y", 10); err != nil || val != 10 {
		t.Errorf("cannot get default value: wanted: %d, got: %d, %v", 10, val, err)
	}
}

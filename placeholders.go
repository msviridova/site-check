package main

import "strings"

// applyPlaceholders заменяет все плейсхолдеры вида {KEY} на значения из мапы.
// Ключи в replacements должны включать точный плейсхолдер (например, "{SITE_TEXT}").
func applyPlaceholders(template string, replacements map[string]string) string {
	if len(replacements) == 0 {
		return template
	}

	args := make([]string, 0, len(replacements)*2)
	for placeholder, value := range replacements {
		args = append(args, placeholder, value)
	}

	return strings.NewReplacer(args...).Replace(template)
}
